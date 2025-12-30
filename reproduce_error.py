import json

def test_none_keys():
    response_body = "null"
    response_json = json.loads(response_body)
    print(f"response_json: {response_json}")
    try:
        print(f"keys: {list(response_json.keys())}")
    except AttributeError as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    test_none_keys()
