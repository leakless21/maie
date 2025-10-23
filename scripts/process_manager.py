#!/usr/bin/env python3
"""
Process Manager for MAIE Development Environment

This script helps identify and optionally kill running MAIE server and worker processes.
It can detect processes by:
- Process name patterns
- Command line arguments
- Port usage
- Worker names in Redis
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    print("Error: psutil is required. Install with: pip install psutil")
    sys.exit(1)

try:
    from redis import Redis
except ImportError:
    print("Error: redis is required. Install with: pip install redis")
    sys.exit(1)


class MAIEProcessManager:
    """Manages MAIE server and worker processes."""
    
    def __init__(self):
        self.api_processes: List[psutil.Process] = []
        self.worker_processes: List[psutil.Process] = []
        self.redis_workers: List[Dict] = []
        
    def find_api_processes(self) -> List[psutil.Process]:
        """Find running API server processes."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                # Look for uvicorn processes with MAIE app
                if ('uvicorn' in cmdline and 
                    'src.api.main:app' in cmdline):
                    processes.append(proc)
                    
                # Look for pixi run api processes
                elif ('pixi' in cmdline and 
                      'run' in cmdline and 
                      'api' in cmdline):
                    processes.append(proc)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
        return processes
    
    def find_worker_processes(self) -> List[psutil.Process]:
        """Find running worker processes."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                # Look for python processes running worker main
                if ('python' in cmdline and 
                    'src/worker/main.py' in cmdline):
                    processes.append(proc)
                    
                # Look for pixi run worker processes
                elif ('pixi' in cmdline and 
                      'run' in cmdline and 
                      'worker' in cmdline):
                    processes.append(proc)
                    
                # Look for RQ worker processes
                elif ('rq' in cmdline and 'worker' in cmdline):
                    processes.append(proc)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
        return processes
    
    def find_processes_by_port(self, port: int = 8000) -> List[psutil.Process]:
        """Find processes using a specific port."""
        processes = []
        
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port and conn.status == 'LISTEN':
                try:
                    proc = psutil.Process(conn.pid)
                    processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        return processes
    
    def get_redis_workers(self) -> List[Dict]:
        """Get active RQ workers from Redis."""
        workers = []
        
        try:
            # Try to connect to Redis (using default settings)
            redis_conn = Redis.from_url('redis://localhost:6379/0', decode_responses=True)
            redis_conn.ping()
            
            # Get worker information
            worker_keys = redis_conn.keys('rq:worker:*')
            for key in worker_keys:
                worker_data = redis_conn.hgetall(key)
                if worker_data:
                    workers.append({
                        'name': worker_data.get('name', 'unknown'),
                        'state': worker_data.get('state', 'unknown'),
                        'last_heartbeat': worker_data.get('last_heartbeat', 'unknown'),
                        'key': key
                    })
                    
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}")
            
        return workers
    
    def scan_all_processes(self):
        """Scan for all MAIE-related processes."""
        print("🔍 Scanning for MAIE processes...")
        
        # Find API processes
        self.api_processes = self.find_api_processes()
        
        # Find worker processes
        self.worker_processes = self.find_worker_processes()
        
        # Find processes by port
        port_processes = self.find_processes_by_port(8000)
        
        # Get Redis workers
        self.redis_workers = self.get_redis_workers()
        
        # Merge port processes with API processes (avoid duplicates)
        for proc in port_processes:
            if proc not in self.api_processes:
                self.api_processes.append(proc)
    
    def display_processes(self):
        """Display found processes in a formatted way."""
        print("\n" + "="*60)
        print("📊 MAIE PROCESS STATUS")
        print("="*60)
        
        # API Processes
        print(f"\n🌐 API Server Processes ({len(self.api_processes)} found):")
        if self.api_processes:
            for proc in self.api_processes:
                try:
                    cmdline = ' '.join(proc.cmdline())
                    print(f"  PID: {proc.pid:>6} | {proc.name():>12} | {cmdline[:80]}...")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"  PID: {proc.pid:>6} | <process info unavailable>")
        else:
            print("  No API server processes found")
        
        # Worker Processes
        print(f"\n⚙️  Worker Processes ({len(self.worker_processes)} found):")
        if self.worker_processes:
            for proc in self.worker_processes:
                try:
                    cmdline = ' '.join(proc.cmdline())
                    print(f"  PID: {proc.pid:>6} | {proc.name():>12} | {cmdline[:80]}...")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"  PID: {proc.pid:>6} | <process info unavailable>")
        else:
            print("  No worker processes found")
        
        # Redis Workers
        print(f"\n🔗 Redis Workers ({len(self.redis_workers)} found):")
        if self.redis_workers:
            for worker in self.redis_workers:
                print(f"  Name: {worker['name']:>20} | State: {worker['state']:>10} | Last Heartbeat: {worker['last_heartbeat']}")
        else:
            print("  No Redis workers found")
        
        print("\n" + "="*60)
    
    def kill_processes(self, force: bool = False):
        """Kill all found processes."""
        signal_type = signal.SIGKILL if force else signal.SIGTERM
        signal_name = "SIGKILL" if force else "SIGTERM"
        
        print(f"\n💀 Killing processes with {signal_name}...")
        
        all_processes = self.api_processes + self.worker_processes
        
        if not all_processes:
            print("  No processes to kill")
            return
        
        killed_count = 0
        for proc in all_processes:
            try:
                proc.send_signal(signal_type)
                print(f"  ✅ Sent {signal_name} to PID {proc.pid}")
                killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"  ❌ Failed to kill PID {proc.pid}: {e}")
        
        print(f"\n📊 Summary: {killed_count}/{len(all_processes)} processes killed")
        
        # Also try to clean up Redis workers
        if self.redis_workers:
            print("\n🧹 Cleaning up Redis workers...")
            try:
                redis_conn = Redis.from_url('redis://localhost:6379/0', decode_responses=True)
                for worker in self.redis_workers:
                    redis_conn.delete(worker['key'])
                    print(f"  ✅ Removed Redis worker: {worker['name']}")
            except Exception as e:
                print(f"  ❌ Failed to clean Redis workers: {e}")
    
    def interactive_kill(self):
        """Interactive process killing with confirmation."""
        all_processes = self.api_processes + self.worker_processes
        
        if not all_processes:
            print("\n✅ No processes found to kill")
            return
        
        print(f"\n⚠️  Found {len(all_processes)} processes to kill:")
        for proc in all_processes:
            try:
                cmdline = ' '.join(proc.cmdline())
                print(f"  - PID {proc.pid}: {cmdline[:60]}...")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"  - PID {proc.pid}: <process info unavailable>")
        
        response = input(f"\n🤔 Kill all {len(all_processes)} processes? [y/N]: ").strip().lower()
        
        if response in ['y', 'yes']:
            force_response = input("💀 Force kill (SIGKILL)? [y/N]: ").strip().lower()
            force = force_response in ['y', 'yes']
            self.kill_processes(force=force)
        else:
            print("❌ Cancelled")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MAIE Process Manager - Identify and kill running server/worker processes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/process_manager.py                    # Show running processes
  python scripts/process_manager.py --kill            # Kill all processes
  python scripts/process_manager.py --kill --force    # Force kill all processes
  python scripts/process_manager.py --interactive     # Interactive kill mode
        """
    )
    
    parser.add_argument(
        '--kill', 
        action='store_true', 
        help='Kill all found processes'
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Use SIGKILL instead of SIGTERM (use with --kill)'
    )
    parser.add_argument(
        '--interactive', '-i', 
        action='store_true', 
        help='Interactive mode - ask before killing'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000, 
        help='Port to check for API server (default: 8000)'
    )
    parser.add_argument(
        '--json', 
        action='store_true', 
        help='Output in JSON format'
    )
    
    args = parser.parse_args()
    
    # Create process manager
    manager = MAIEProcessManager()
    
    # Scan for processes
    manager.scan_all_processes()
    
    if args.json:
        # JSON output
        result = {
            'api_processes': [
                {
                    'pid': proc.pid,
                    'name': proc.name(),
                    'cmdline': proc.cmdline()
                } for proc in manager.api_processes
            ],
            'worker_processes': [
                {
                    'pid': proc.pid,
                    'name': proc.name(),
                    'cmdline': proc.cmdline()
                } for proc in manager.worker_processes
            ],
            'redis_workers': manager.redis_workers
        }
        print(json.dumps(result, indent=2))
        return
    
    # Display processes
    manager.display_processes()
    
    # Handle killing
    if args.kill:
        if args.interactive:
            manager.interactive_kill()
        else:
            manager.kill_processes(force=args.force)
    elif args.interactive:
        manager.interactive_kill()


if __name__ == '__main__':
    main()
