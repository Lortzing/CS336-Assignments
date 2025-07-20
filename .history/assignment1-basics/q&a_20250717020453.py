def problem_unicode1() -> None:
    print(chr(0))
    print(chr(0).__repr__())
    
    # run in Python interpreter
    # >>> chr(0)
    # >>> print(chr(0))
    # >>> "this is a test" + chr(0) + "string"
    # >>> print("this is a test" + chr(0) + "string")
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])