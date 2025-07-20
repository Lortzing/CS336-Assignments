from functools import wraps


def parting_line(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        print("-" * 40)
        print(fun.__name__+" starts")
        print("-" * 40)
        fun(*args, **kwargs)
        print("-" * 40)
    return wrapper

def problem_unicode1() -> None:
    print(chr(0))
    print(chr(0).__repr__())
    
    # run in Python interpreter
    # >>> chr(0)
    # >>> print(chr(0))
    # >>> "this is a test" + chr(0) + "string"
    # >>> print("this is a test" + chr(0) + "string")

def problem_unicode2() -> None:
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
    try:
        print(decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8")))
    except Exception as e:
        print("Error in process `hello! こんにちは!`")
        print(f"{'hello! こんにちは!'.encode("utf-8")=}")
    
    try:
        print(bytes([255, 235]).decode("utf-8"))
    except Exception as e:
        print("Error in process `bytes([255, 235])`")
        print(e)
    
    
if __name__ == '__main__':
    problem_unicode2()