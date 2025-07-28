# WriteUp

## Problem (unicode1): Understanding Unicode (1 point)

### (a) What Unicode character does chr(0) return?
Answer: chr(0) returns the NULL character.

### (b) How does this character’s string representation (`__repr__()`) differ from its printed representation?

Answer: The `__repr__()` of chr(0) shows '\\x00', but printing it produces an invisible, non-printable character.

### (c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")

Answer:
```python
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

## Problem (unicode2): Unicode Encodings (3 points)

### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

Answer:
1. The 256-length byte vocabulary is much more manageable to deal with. 
2. And the vocabulary on UTF-16 or UTF-32 would be prohibitively large and sparse.
3. UTF-8 is the dominant encoding for the Internet.

### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
Answer:
- Example: `"hello! こんにちは!"`
- Reason: It's incorrect to decode from byte string one integers every time.

### (c) Give a two byte sequence that does not decode to any Unicode character(s).

Answer:
- Example: `bytes([255, 235])`
- This byte sequence is not a valid UTF-8 encoding—`0xff` is not a valid start byte in UTF-8, so decoding this will raise a UnicodeDecodeError.


