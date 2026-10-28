from hello import greet

def test_greet_custom_name():
    result = greet("DevOps")
    assert result == "Hello, DevOps from CaneCCS Docker Container!"
    print(f"\n[PASSED TEST CASE]: {result}")
