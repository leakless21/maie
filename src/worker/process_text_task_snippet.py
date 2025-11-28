
def process_text_task(task_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pipeline function for text-only processing (summary/enhancement).
    
    Skips audio preprocessing and ASR stages, directly invoking the LLM.
    
    Args:
        task_params: Dictionary containing:
            - text: Input text to process
            - features: List of features (summary, clean_transcript)
            - template_id: Template ID for summary
            - redis_host, redis_port, redis_db: Redis connection params
            
    Returns:
        Dictionary containing processing results
    """
    job = get_current_job()
    job_id = job.id if job else "unknown"
    task_key = f"task:{job_id}"
    
    # Bind correlation ID
    _cid = task_params.get("correlation_id")
    if _cid:
        try:
            bind_correlation_id(str(_cid))
        except Exception:
            pass

    # Extract parameters
    text = task_params.get("text", "")
    features = task_params.get("features", ["summary"])
    template_id = task_params.get("template_id")
    
    # Connect to Redis results DB
    redis_host = task_params.get("redis_host", "localhost")
    redis_port = task_params.get("redis_port", 6379)
    redis_db = task_params.get("redis_db", 1)
    redis_conn = (
        Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)
        if job
        else None
    )
    
    start_time = time.time()
    llm_model = None
    version_metadata: Optional[Dict[str, Any]] = None
    
    try:
        logger.info(
            "Starting text processing",
            task_id=job_id,
            text_length=len(text),
            features=features
        )
        
        if redis_conn:
            _update_status(redis_conn, task_key, TaskStatus.PROCESSING_LLM)
            
        # Validate input
        if not text or not isinstance(text, str) or not text.strip():
             raise LLMProcessingError(
                message="Input text is empty or invalid",
                details={"text_length": len(text) if text else 0}
             )

        # Load LLM
        logger.info("Loading LLM model", task_id=job_id)
        
        # Clear GPU cache if local vLLM
        if (
            TORCH_AVAILABLE
            and torch is not None
            and has_cuda()
            and settings.llm_backend == LlmBackendType.LOCAL_VLLM
        ):
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
                
        from src.processors.llm import LLMProcessor
        llm_model = LLMProcessor()
        llm_model._load_model()
        
        # Process features
        clean_transcript = text
        structured_summary = None
        
        # Enhancement
        if "clean_transcript" in features:
            try:
                enhanced_result = llm_model.enhance_text(text)
                if enhanced_result.get("enhancement_applied", False):
                    clean_transcript = enhanced_result["enhanced_text"]
                logger.info("Text enhancement complete", task_id=job_id)
            except Exception as e:
                logger.warning("Text enhancement failed, using original text", error=str(e))
                
        # Summary
        if "summary" in features:
            if not template_id:
                raise LLMProcessingError("template_id required for summary")
                
            try:
                summary_result = llm_model.generate_summary(
                    transcript=clean_transcript, 
                    template_id=template_id
                )
                if summary_result.get("summary"):
                    structured_summary = summary_result["summary"]
                else:
                    raise LLMProcessingError(f"Summary generation failed: {summary_result.get('error')}")
            except Exception as e:
                logger.error("Summary generation failed", error=str(e))
                raise LLMProcessingError(f"Summary generation failed: {str(e)}")
        
        # Collect metadata
        try:
            # Create dummy ASR metadata for compatibility
            asr_metadata = {
                "name": "text_input",
                "model_variant": "none",
                "model_path": "none", 
                "checkpoint_hash": "none",
                "compute_type": "none",
                "decoding_params": {}
            }
            version_metadata = get_version_metadata(asr_metadata, llm_model)
        except Exception:
            version_metadata = None
            
        # Calculate metrics
        processing_time = time.time() - start_time
        metrics = {
            "input_duration_seconds": 0.0, # Text input has no duration
            "processing_time_seconds": processing_time,
            "rtf": 0.0,
            "vad_coverage": 0.0,
            "asr_confidence_avg": 1.0,
            "transcription_length": len(text)
        }
        
        # Prepare results
        result = {
            "versions": version_metadata,
            "metrics": metrics,
            "results": {
                "raw_transcript": text,
                "clean_transcript": clean_transcript
            }
        }
        
        if structured_summary:
            result["results"]["summary"] = structured_summary
            
        # Update status to COMPLETE
        if redis_conn:
            _update_status(
                redis_conn,
                task_key,
                TaskStatus.COMPLETE,
                {
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "versions": version_metadata,
                    "metrics": metrics,
                    "results": result["results"]
                }
            )
            
        logger.info("Text processing completed successfully", task_id=job_id)
        
        try:
            clear_correlation_id()
        except Exception:
            pass
            
        return result

    except Exception as e:
        logger.error(f"Text processing failed: {e}", task_id=job_id)
        
        if redis_conn:
            error_details = {
                "status": TaskStatus.FAILED.value,
                "error": str(e),
                "stage": "text_processing",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            redis_conn.hset(task_key, mapping=error_details)
            
        try:
            clear_correlation_id()
        except Exception:
            pass
            
        return {
            "status": "error",
            "error": {"message": str(e)},
            "task_id": job_id
        }
    finally:
        # Unload LLM
        if llm_model:
            try:
                if hasattr(llm_model, "unload"):
                    llm_model.unload()
                if TORCH_AVAILABLE and torch is not None and has_cuda():
                    torch.cuda.empty_cache()
            except Exception:
                pass
