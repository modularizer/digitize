# digitize
A natural language number string normalizer

## Install
```bash
pip install -e .
```

## Python Usage
```python
from digitize import digitize

print(digitize("two point eight multiplied by twenty two point 7", config="nomath")) # 2.8*22.7
print(digitize("two point eight multiplied by twenty two point 7")) # 63.56
```

## CLI usage
```bash
# call directly
digitize "I have 5 dozen donuts" 
# "I have 60 donuts"

# or via pipe
echo "I have 3 pairs of shoes" | digitize 
# "I have 6 shoes"

# help
digitize -h

# demo
digitize -m demo
```

## NOTE
In most projects I write readable code with a few tests. In this project the code is very difficult to follow, but there are 200 tests. It is a tradeoff...