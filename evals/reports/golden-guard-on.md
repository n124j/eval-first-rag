# Evaluation report

| Field | Value |
| --- | --- |
| model | qwen2.5-7b-instruct-1m |
| hardware | AMD Ryzen 9 3900X, 128 GB RAM, Radeon RX 6600 XT 8 GB; served by LM Studio (GPU offload setting not recorded) |
| date | 2026-10-20 |
| k | 3 |
| min_dense_score | 0.67 |
| items | 60 |

## Metrics

| Metric | Value |
| --- | --- |
| recall_at_k | 0.8800 |
| mrr | 0.8967 |
| citation_validity_rate | 0.3000 |
| correct_refusal_rate | 0.8000 |
| over_refusal_rate | 0.1800 |
| latency_p50_s | 4.373 |
| latency_p95_s | 7.113 |
| prompt_tokens_total | 76449 |
| completion_tokens_total | 2970 |

## Failures

- g001 (single): When a value is already in a sorted list, where does bisect_left place the insertion point relative to the existing equal entries? | failures: invalid citations | retrieved sources: bisect | cited: bisect#001
- g002 (single): In a heap managed by heapq, which element is always stored at index 0? | failures: invalid citations | retrieved sources: heapq | cited: heapq#000
- g004 (single): How does shutil.copy2 differ from shutil.copy? | failures: invalid citations | retrieved sources: shutil | cited: shutil#006
- g005 (single): What is the difference between statistics.stdev and statistics.pstdev? | failures: invalid citations | retrieved sources: statistics | cited: statistics#014
- g007 (single): What does functools.partial return when I give it a function and some arguments? | failures: invalid citations | retrieved sources: functools | cited: functools#021
- g008 (single): What happens when I assign to a field of a dataclass declared with frozen=True? | failures: invalid citations | retrieved sources: dataclasses | cited: dataclasses#020
- g011 (single): Which characters are included in string.punctuation? | failures: invalid citations | retrieved sources: string, re | cited: string#000
- g015 (single): Which re flag makes a regular expression match letters regardless of case? | failures: invalid citations | retrieved sources: re | cited: re#013
- g016 (single): How do I duplicate a nested list so that changing the inner lists of the duplicate leaves the original untouched? | failures: recall 0 | retrieved sources: itertools, collections, random | cited: itertools#017
- g017 (single): How can I break a long list into consecutive groups of three items, where the last group may be smaller? | failures: refused answerable question | retrieved sources: itertools, heapq | cited: none
- g019 (single): My file search with a '**' wildcard only looks one folder deep instead of descending into every subfolder; what am I missing? | failures: refused answerable question | retrieved sources: glob, bisect, re | cited: none
- g020 (single): How do I get a scratch folder that is deleted automatically when I am finished with it? | failures: invalid citations | retrieved sources: tempfile | cited: tempfile#003
- g021 (single): How do I compute a SHA-256 checksum of some bytes in Python? | failures: invalid citations | retrieved sources: hashlib | cited: hashlib#000
- g022 (single): How do I turn a text value like '2024-03-15' into a date object? | failures: invalid citations | retrieved sources: datetime | cited: datetime#019
- g023 (single): How do I test whether two floating-point numbers are nearly equal rather than exactly equal? | failures: invalid citations | retrieved sources: decimal, math | cited: math#002
- g024 (single): How should I pass values typed in by a user into an SQL statement so they cannot inject SQL? | failures: invalid citations | retrieved sources: sqlite3 | cited: sqlite3#035
- g025 (single): How do I send my program's log messages to a file instead of the terminal? | failures: refused answerable question | retrieved sources: logging, shutil | cited: none
- g026 (single): How do I run an external program and get back the text it wrote to standard output and standard error? | failures: invalid citations | retrieved sources: subprocess | cited: none
- g027 (single): How do I stop two threads from modifying the same shared variable at the same moment? | failures: invalid citations | retrieved sources: threading | cited: threading#022
- g028 (single): How do I make one of my unit tests be skipped only when some condition holds? | failures: invalid citations | retrieved sources: unittest | cited: unittest#015
- g029 (single): How should I write a type hint for a parameter that may be an int or may be None? | failures: invalid citations | retrieved sources: typing | cited: none
- g030 (single): Is there a tidy way to ignore one particular exception raised inside a with block without writing try/except? | failures: invalid citations | retrieved sources: contextlib | cited: contextlib#007
- g032 (single): My recursive function recomputes the same results over and over and is very slow; is there a built-in way to remember the results of earlier calls? | failures: recall 0, refused answerable question | retrieved sources: heapq, threading, typing | cited: none
- g034 (single): When I split the file name '.bashrc' into a base name and an extension, why is the extension empty? | failures: invalid citations | retrieved sources: os.path | cited: none
- g035 (single): Why do I get a TypeError when I check whether a timestamp that carries a time zone is earlier than one that does not? | failures: invalid citations | retrieved sources: datetime | cited: datetime#038
- g036 (single): The help text of my command-line tool shows the value placeholder of an option in all capital letters; how can I change what is shown there? | failures: invalid citations | retrieved sources: argparse | cited: none
- g037 (single): Why does my class decorated with @dataclass raise ValueError when I give a field an empty list as its default? | failures: invalid citations | retrieved sources: dataclasses | cited: dataclasses#024
- g038 (single): My script hangs forever when the child program it started prints a lot of output and I wait for the child to exit; what is going on? | failures: invalid citations | retrieved sources: subprocess | cited: subprocess#027
- g039 (single): How can I let my program exit even though a background worker is still looping forever? | failures: recall 0, refused answerable question | retrieved sources: contextlib | cited: none
- g042 (multi): How can I load the rows of a comma-separated text file into an SQLite table? | failures: refused answerable question | retrieved sources: sqlite3 | cited: none
- g044 (multi): How do I move a date forward by one month, falling back to the last day of the next month when the same day number does not exist there (for example from January 31)? | failures: invalid citations | retrieved sources: datetime, calendar | cited: none
- g045 (multi): How do I combine all the integers in a list into one value with bitwise OR, without writing a lambda? | failures: recall 0, refused answerable question | retrieved sources: itertools, random, heapq | cited: none
- g046 (multi): How do I count how often each pair of adjacent elements occurs in a sequence? | failures: invalid citations | retrieved sources: itertools, collections | cited: none
- g048 (multi): How do I turn the query string of a URL into a dictionary and output that dictionary as JSON text? | failures: invalid citations | retrieved sources: json, urllib.parse | cited: urllib.parse#023, json#005
- g049 (multi): How do I pick a random member of an Enum class? | failures: refused answerable question | retrieved sources: enum | cited: none
- g050 (multi): How do I get the three tuples with the highest value in their second position from a large list, without sorting the whole list and without writing a lambda for the key? | failures: refused answerable question | retrieved sources: itertools, heapq | cited: none
- g053 (unanswerable): How do I use a pool of worker processes to apply a function to every item of a list in parallel? | failures: answered unanswerable question | retrieved sources: threading, random, heapq | cited: threading#031
- g059 (unanswerable): How do I read a CSV file into a pandas DataFrame and group the rows by a column? | failures: answered unanswerable question | retrieved sources: csv | cited: none
