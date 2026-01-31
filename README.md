# digitize
A natural language number string normalizer for Python and Typescript.

# DISCLAIMER!!
* written for fun, not for production
* Definitely not optimized for efficiency
* Super dumb and way too smart
* Decently tested (>200 tests of various use cases)
* Extremely configurable

## Install
```bash
pip install modularizer-digitize
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

## NOTES
**TL;DR;** trust this project only as much as you trust the tests.


| prompt                                             | output                                                       | params                                                       |
| -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| i have one hundred and one dalmatians              | i have 101 dalmatians                                        | {}                                                           |
| one million and two hundred and thirty four        | 1,000,234                                                    | {'use_commas': True}                                         |
| Traffic hit 10K requests per second.               | Traffic hit 10000 requests per second.                       | {}                                                           |
| one hundred and first place                        | 101st place                                                  | {}                                                           |
| we shipped on the 11th hour                        | we shipped on the 11th hour                                  | {}                                                           |
| she pinged me twice                                | she pinged me 2 times                                        | {}                                                           |
| he tried two times and failed                      | he tried 2x and failed                                       | {'fmt_rep': '%nx', 'rep_signifiers': 'times?'}               |
| third time was the charm                           | 3-thx was the charm                                          | {'fmt_nth_time': '%n-thx', 'rep_signifiers': 'times?'}       |
| the second attempt at which it worked              | the 2nd time it worked                                       | {'fmt_rep': '%nx', 'rep_signifiers': '(?:time|occurence|i... |
| first place, two times, and 3M users               | [NUM=1,ORD=st,OG=first] place, [NUM=2,REP=times,OG=two ti... | {'config': 'token'}                                          |
| one hundred and thirty-five                        | 135                                                          | {}                                                           |
| one thousand and one                               | 1001                                                         | {}                                                           |
| one thousand and one nights                        | 1001 nights                                                  | {}                                                           |
| i have one hundred and one dalmatians              | i have 101 dalmatians                                        | {}                                                           |
| she counted one, and then two, and then three      | she counted 1, and then 2, and then 3                        | {}                                                           |
| one hundred and twenty three thousand and four     | 123004                                                       | {}                                                           |
| one million and two hundred and thirty four        | 1000234                                                      | {}                                                           |
| one hundred and                                    | 100 and                                                      | {}                                                           |
| and one hundred                                    | and 100                                                      | {}                                                           |
| he whispered one hundred and thirty five times     | he whispered 135x                                            | {'fmt_rep': '%nx'}                                           |
| one hundred 33 dalmatians                          | 133 dalmatians                                               | {}                                                           |
| one thousand 2 nights                              | 1002 nights                                                  | {}                                                           |
| one million two hundred and thirty four            | 1000234                                                      | {}                                                           |
| one hundred and 33                                 | 133                                                          | {}                                                           |
| one thousand and 2                                 | 1002                                                         | {}                                                           |
| one hundred and thirty five, exactly               | 135, exactly                                                 | {}                                                           |
| he said one hundred and one.                       | he said 101.                                                 | {}                                                           |
| i have (one hundred and one) dalmatians            | i have (101) dalmatians                                      | {}                                                           |
| one hundred and one  nights                        | 101  nights                                                  | {}                                                           |
| one hundred and thirty-five-six                    | 129                                                          | {}                                                           |
| one hundred and thirty-five and six and seven      | 135 and 6 and 7                                              | {}                                                           |
| one and two hundred                                | 1 and 200                                                    | {}                                                           |
| one hundred and one and two                        | 101 and 2                                                    | {}                                                           |
| one hundred and thirty five                        | 135                                                          | {'use_commas': True}                                         |
| one hundred and thirty five thousand               | 135,000                                                      | {'use_commas': True}                                         |
| one hundred and thirty five pigs                   | [135] pigs                                                   | {'fmt': '[%n]'}                                              |
| he saw one hundred and thirty five birds           | he saw  birds                                                | {'fmt': ''}                                                  |
| one million and two hundred and thirty four        | 1,000,234                                                    | {'use_commas': True}                                         |
| balance: 0,000 and one                             | balance: 0,000 and 1                                         | {}                                                           |
| he saw one hundred and one birds                   | he saw <101> birds                                           | {'fmt': '<%n>'}                                              |
| he saw one hundred and one birds                   | he saw  birds                                                | {'fmt': ''}                                                  |
| someone said one hundred and nothing else          | someone said 100 and nothing else                            | {}                                                           |
| stone and one hundred and one                      | stone and 101                                                | {}                                                           |
| one and one is two                                 | 1 and 1 is 2                                                 | {}                                                           |
| one hundred and thirty-five and six                | 135 and 6                                                    | {}                                                           |
| one hundred and thirty five-six                    | 129                                                          | {}                                                           |
|                                                    |                                                              | {}                                                           |
|                                                    |                                                              | {}                                                           |
| one... and then one hundred and one!!!             | 1... and then 101!!!                                         | {}                                                           |
| We raised 3M dollars.                              | We raised 3000000 dollars.                                   | {}                                                           |
| Traffic hit 10K requests per second.               | Traffic hit 10000 requests per second.                       | {}                                                           |
| Storage is 2G and climbing.                        | Storage is 2000000000 and climbing.                          | {}                                                           |
| Budget: 2.5M for phase one.                        | Budget: 2500000 for phase 1.                                 | {}                                                           |
| Big money: 1T reasons to care.                     | Big money: 1000000000000 reasons to care.                    | {}                                                           |
| Rare air: 1P possibilities.                        | Rare air: 1000000000000000 possibilities.                    | {}                                                           |
| We raised 3m dollars.                              | We raised 3m dollars.                                        | {}                                                           |
| Traffic hit 10k requests per second.               | Traffic hit 10000 requests per second.                       | {}                                                           |
| He wrote 2g on the napkin.                         | He wrote 2g on the napkin.                                   | {}                                                           |
| I have 10K and one dreams.                         | I have 10000 and 1 dreams.                                   | {}                                                           |
| Deploy to 3M users and keep two backups.           | Deploy to 3000000 users and keep 2 backups.                  | {}                                                           |
| Worth 10K, maybe 2.5M, never 3m.                   | Worth 10000, maybe 2500000, never 3m.                        | {}                                                           |
| Edge: 10K.                                         | Edge: 10,000.                                                | {'use_commas': True}                                         |
| Edge: 10K, 2.5M, and 1G.                           | Edge: 10000, 2500000, and 1000000000.                        | {}                                                           |
| Edge: 10K.                                         | Edge: 10000.                                                 | {'use_commas': False}                                        |
| Edge: 10K.                                         | Edge: 10,000.                                                | {'use_commas': True}                                         |
| not a suffix: 10Km and 2.5Ms                       | not a suffix: 10Km and 2.5Ms                                 | {}                                                           |
| we raised 3M dollars and spent 2K                  | we raised 3,000,000 dollars and spent 2,000                  | {'use_commas': True}                                         |
| I have $10K and one dreams.                        | I have $10000 and 1 dreams.                                  | {}                                                           |
| first place                                        | 1st place                                                    | {}                                                           |
| second attempt                                     | 2nd time                                                     | {}                                                           |
| third try                                          | 3rd time                                                     | {}                                                           |
| fourth wall                                        | 4th wall                                                     | {}                                                           |
| twenty-first century                               | 21st century                                                 | {}                                                           |
| one hundred seventy-second                         | 172nd                                                        | {}                                                           |
| one hundred and first                              | 101st                                                        | {}                                                           |
| 11th hour                                          | 11th hour                                                    | {}                                                           |
| first prize                                        | [1st] prize                                                  | {'fmt_ordinal': '[%n%o]'}                                    |
| second prize                                       |  prize                                                       | {'fmt_ordinal': ''}                                          |
| once                                               | 1 time                                                       | {}                                                           |
| twice                                              | 2x                                                           | {'fmt_rep': '%nx'}                                           |
| thrice                                             | 3x                                                           | {'fmt_rep': '%nx'}                                           |
| one time                                           | 1 time                                                       | {}                                                           |
| two times                                          | 2x                                                           | {'fmt_rep': '%nx'}                                           |
| 1 time                                             | 1x                                                           | {'fmt_rep': '%nx'}                                           |
| 3 times                                            | 3x                                                           | {'fmt_rep': '%nx'}                                           |
| he tried two times                                 | he tried 2x                                                  | {'fmt_rep': '%nx'}                                           |
| first time                                         | 1st time                                                     | {}                                                           |
| second time                                        | 2nd time                                                     | {}                                                           |
| twenty-first time                                  | 21stx                                                        | {'fmt_nth_time': '%n%ox'}                                    |
| one hundredth time                                 | 100thx                                                       | {'fmt_nth_time': '%n%ox'}                                    |
| five hundredth time                                | 500th occurence                                              | {'fmt_nth_time': '%n%o occurence'}                           |
| twice                                              | 2×                                                           | {'fmt_rep': '%n×'}                                           |
| third time                                         | <3rd x>                                                      | {'fmt_nth_time': '<%n%o x>'}                                 |
| first and second time                              | 1st and 2nd time                                             | {}                                                           |
| once and twice                                     | 1x and 2x                                                    | {'fmt_rep': '%nx'}                                           |
| one time and two                                   | 1x and 2                                                     | {'fmt_rep': '%nx'}                                           |
| Chapter IV                                         | Chapter 4                                                    | {'support_roman': True}                                      |
| Henry VIII                                         | Henry 8                                                      | {'support_roman': True}                                      |
| Year MMXXIII                                       | Year 2023                                                    | {'support_roman': True}                                      |
| Section IX                                         | Section 9                                                    | {'support_roman': True}                                      |
| Part iii                                           | Part 3                                                       | {'support_roman': True}                                      |
| Volume XL                                          | Volume 40                                                    | {'support_roman': True}                                      |
| Not a roman numeral: MMXIXI                        | Not a roman numeral: MMXIXI                                  | {'support_roman': True}                                      |
| Mixed: Chapter IV and Section X                    | Mixed: Chapter 4 and Section 10                              | {'support_roman': True}                                      |
| Without support: Chapter IV                        | Without support: Chapter IV                                  | {'support_roman': False}                                     |
| negative five                                      | -5                                                           | {}                                                           |
| positive ten                                       | +10                                                          | {}                                                           |
| neg 3                                              | -3                                                           | {}                                                           |
| pos 7                                              | +7                                                           | {}                                                           |
| - 100                                              | -100                                                         | {}                                                           |
| + 50                                               | +50                                                          | {}                                                           |
| negative one hundred                               | -100                                                         | {}                                                           |
| positive two thousand                              | +2000                                                        | {}                                                           |
| it was negative five degrees                       | it was -5 degrees                                            | {}                                                           |
| score: + 10                                        | score: +10                                                   | {}                                                           |
| no sign 5                                          | no sign 5                                                    | {}                                                           |
| negative-five                                      | negative-5                                                   | {}                                                           |
| positive-six                                       | positive-6                                                   | {}                                                           |
| negative five and positive six                     | -5 and +6                                                    | {}                                                           |
| neg 5 and pos 6                                    | -5 and +6                                                    | {}                                                           |
| value is - 5                                       | value is -5                                                  | {}                                                           |
| value is + 5                                       | value is +5                                                  | {}                                                           |
| negative one hundred and one                       | -101                                                         | {}                                                           |
| positive one hundred and one                       | +101                                                         | {}                                                           |
| negative 5 and 6                                   | -5 and 6                                                     | {}                                                           |
| negative 5 and negative 6                          | -5 and -6                                                    | {}                                                           |
| disabled: negative five                            | disabled: negative 5                                         | {'parse_signs': False}                                       |
| zero point five                                    | 0.5                                                          | {}                                                           |
| point five                                         | 0.5                                                          | {}                                                           |
| two point zero five                                | 2.05                                                         | {}                                                           |
| one hundred and one point two                      | 101.2                                                        | {}                                                           |
| negative zero point five                           | -0.5                                                         | {}                                                           |
| we shipped zero point five days early              | we shipped 0.5 days early                                    | {}                                                           |
| we shipped oh point five days early                | we shipped 0.5 days early                                    | {}                                                           |
| we shipped oh point oh five seven days early       | we shipped 0.057 days early                                  | {}                                                           |
| we shipped five thousand point two days early      | we shipped 5000.2 days early                                 | {}                                                           |
| the hundredth time                                 | the 100th time                                               | {}                                                           |
| the fifth time                                     | the 5th time                                                 | {}                                                           |
| the fourth amendment                               | the 4th amendment                                            | {}                                                           |
| one hundredth                                      | 1/100                                                        | {'do_simple_evals': False}                                   |
| one hundredth                                      | 0.01                                                         | {}                                                           |
| 1 hundredth                                        | 1/100                                                        | {'do_simple_evals': False}                                   |
| 1 hundredth                                        | 0.01                                                         | {}                                                           |
| 1 100th                                            | 0.01                                                         | {}                                                           |
| two hundredths                                     | 0.02                                                         | {}                                                           |
| five thousandths                                   | 0.005                                                        | {}                                                           |
| five point 8 millionth                             | 0.0000058                                                    | {}                                                           |
| one third                                          | 1/3                                                          | {}                                                           |
| a third                                            | 1/3                                                          | {}                                                           |
| two thirds                                         | 2/3                                                          | {}                                                           |
| negative 5 thirds                                  | -5/3                                                         | {}                                                           |
| two halves                                         | 1                                                            | {}                                                           |
| half                                               | 1/2                                                          | {'do_simple_evals': False}                                   |
| half                                               | 0.5                                                          | {}                                                           |
| one over                                           | 1 over                                                       | {}                                                           |
| one over five                                      | 1/5                                                          | {'do_simple_evals': False}                                   |
| one over five                                      | 0.2                                                          | {}                                                           |
| two divided by three                               | 2/3                                                          | {}                                                           |
| five times                                         | 5 times                                                      | {}                                                           |
| five times ten                                     | 5*10                                                         | {'do_simple_evals': False}                                   |
| five times ten                                     | 50                                                           | {}                                                           |
| two point eight multiplied by twenty two point 7   | 2.8*22.7                                                     | {'do_simple_evals': False}                                   |
| two point eight multiplied by twenty two point 7   | 63.56                                                        | {}                                                           |
| five occurences of 10                              | 50                                                           | {}                                                           |
| seven of nine                                      | 7/9                                                          | {}                                                           |
| 6 into 11                                          | 6/11                                                         | {}                                                           |
| seven point five plus 8                            | 15.5                                                         | {}                                                           |
| two to the power of 3                              | 2**3                                                         | {'do_simple_evals': False}                                   |
| two to the power of 3                              | 2^3                                                          | {'power': '^', 'do_simple_evals': False}                     |
| two to the third power                             | 8                                                            | {'power': '^'}                                               |
| two to the third                                   | 8                                                            | {'power': '^'}                                               |
| two to the five hundredth power                    | 2^500                                                        | {'power': '^', 'do_simple_evals': False}                     |
| two to the one hundred twenty-seventh power        | 2^127                                                        | {'power': '^', 'do_simple_evals': False}                     |
| two to the one hundred twenty-seventh power        | 170141183460469231731687303715884105728                      | {'power': '^'}                                               |
| two to the one thousand twenty-seventh power       | 143815450788985272618344415263121978689438158315384525818... | {'power': '^'}                                               |
| square root of 5                                   | 5^(1/2)                                                      | {'power': '^'}                                               |
| square root of 5                                   | 2.236                                                        | {'power': '^', 'do_fraction_evals': True, 'res': 3}          |
| square root of 5                                   | 2.2360679775                                                 | {'power': '^', 'do_fraction_evals': True, 'res': 10}         |
| 5th root of 32                                     | 32**(1/5)                                                    | {}                                                           |
| six e five                                         | 600000                                                       | {}                                                           |
| 6 e -5                                             | 0.00006                                                      | {}                                                           |
| six times ten to the fifth                         | 600000                                                       | {'power': '^', 'mult': ' x '}                                |
| 6*10^5                                             | 600000                                                       | {}                                                           |
| 6*10**(5)                                          | 6*(10**(5))                                                  | {'do_simple_evals': False}                                   |
| one and a half                                     | 1.5                                                          | {}                                                           |
| one and a third                                    | 4/3                                                          | {}                                                           |
| one and a third                                    | 1.333                                                        | {'res': 3}                                                   |
| one and two thirds                                 | 1.667                                                        | {'res': 3}                                                   |
| one and two thirds                                 | 1.6667                                                       | {'res': 4}                                                   |
| one point five                                     | 1.5                                                          | {'res': 4}                                                   |
| one and a half                                     | 1.5                                                          | {'res': 4}                                                   |
| a day and a half                                   | 1.5 days                                                     | {}                                                           |
| a day and a half                                   | 1.5 days                                                     | {}                                                           |
| a day and a half an hour                           | a day and 1/2 an hour                                        | {'do_simple_evals': False}                                   |
| five days and a half                               | 5.5 days                                                     | {}                                                           |
| a dozen eggs                                       | 12 eggs                                                      | {}                                                           |
| five dozen donuts                                  | 5*12 donuts                                                  | {'do_simple_evals': False}                                   |
| five dozen donuts                                  | 60 donuts                                                    | {}                                                           |
| 8 sets of 3 cds                                    | 8*3 cds                                                      | {'do_simple_evals': False}                                   |
| 8 sets of 3 cds                                    | 24 cds                                                       | {}                                                           |
| 8 hours                                            | 28800 s                                                      | {'config': 'units'}                                          |
| 8hours                                             | 28800s                                                       | {'config': 'units'}                                          |
| 8hr                                                | 28800s                                                       | {'config': 'units'}                                          |
| 8hr and 5min                                       | 29100s                                                       | {'config': 'units'}                                          |
| 8hr5min                                            | 29100s                                                       | {'config': 'units'}                                          |
| half of an hour after sunrise                      | 30 minutes after sunrise                                     | {'units': Unit(key=hour, value=60, ...)}                     |
| an hour and a half after sunset                    | an hour and 0.5 after sunset                                 | {}                                                           |
| an hour and a half after sunset                    | 1.5 hours after sunset                                       | {'units': Unit(key=hour, value=1, ...)}                      |
| an hour and a half after sunset                    | 90 minutes after sunset                                      | {'units': Unit(key=hour, value=60, ...)}                     |
| five and two thirds hours after noon               | 17/3 hours after noon                                        | {}                                                           |
| five and two thirds hours after noon               | 340 minutes after noon                                       | {'units': Unit(key=hour, value=60, ...)}                     |
| five minutes and two thirds hours after noon       | 45 minutes after noon                                        | {'units': Unit(key=hour, value=60, ...)}                     |
| an hour and 22 minutes after noon                  | 82 minutes after noon                                        | {'units': Unit(key=hour, value=60, ...)}                     |
| an hour and 22 minutes and 43 seconds after noon   | 4963 s after noon                                            | {'units': UnitGroup(s)}                                      |
| an hour and 23 minutes minus 17 seconds after noon | 4963 s after noon                                            | {'units': UnitGroup(s)}                                      |
| 5 seconds less than a minute                       | 55 s                                                         | {'units': UnitGroup(s)}                                      |
| an hour and a half less than two hours             | 0.5 hours                                                    | {'units': Unit(key=hour, value=1, ...)}                      |
| 5.5 minutes less than two hours                    | 114.5 minutes                                                | {'units': Unit(key=hour, value=60, ...)}                     |
| 5 minutes and 35 seconds less than two hours       | 6865 s                                                       | {'units': UnitGroup(s)}                                      |