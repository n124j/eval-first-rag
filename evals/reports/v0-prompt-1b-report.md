# Evaluation report

| Field | Value |
| --- | --- |
| model | llama-3.2-1b-instruct |
| hardware | AMD Ryzen 9 3900X, 128 GB RAM, Radeon RX 6600 XT 8 GB; served by LM Studio (GPU offload setting not recorded) |
| date | 2026-10-20 |
| k | 5 |
| items | 60 |

## Metrics

| Metric | Value |
| --- | --- |
| recall_at_k | 0.9000 |
| mrr | 0.9017 |
| citation_validity_rate | 0.0000 |
| correct_refusal_rate | 0.0000 |
| latency_p50_s | 1.966 |
| latency_p95_s | 5.525 |
| prompt_tokens_total | 103293 |
| completion_tokens_total | 8527 |

## Failures

- g001 (single): When a value is already in a sorted list, where does bisect_left place the insertion point relative to the existing equal entries? | failures: invalid citations | retrieved sources: bisect | cited: none
- g002 (single): In a heap managed by heapq, which element is always stored at index 0? | failures: invalid citations | retrieved sources: heapq | cited: none
- g003 (single): What kind of string does secrets.token_hex return? | failures: invalid citations | retrieved sources: secrets, re | cited: none
- g004 (single): How does shutil.copy2 differ from shutil.copy? | failures: invalid citations | retrieved sources: shutil | cited: none
- g005 (single): What is the difference between statistics.stdev and statistics.pstdev? | failures: invalid citations | retrieved sources: statistics, random | cited: none
- g006 (single): What does textwrap.dedent do to a block of text? | failures: invalid citations | retrieved sources: textwrap, argparse | cited: none
- g007 (single): What does functools.partial return when I give it a function and some arguments? | failures: invalid citations | retrieved sources: functools | cited: none
- g008 (single): What happens when I assign to a field of a dataclass declared with frozen=True? | failures: invalid citations | retrieved sources: dataclasses | cited: none
- g009 (single): What is the Decimal quantize method used for? | failures: invalid citations | retrieved sources: decimal | cited: none
- g010 (single): Which function in the uuid module generates a random UUID? | failures: invalid citations | retrieved sources: uuid, random | cited: none
- g011 (single): Which characters are included in string.punctuation? | failures: invalid citations | retrieved sources: string, re, urllib.parse | cited: none
- g012 (single): What does operator.itemgetter return when it is created with several items? | failures: invalid citations | retrieved sources: operator, heapq, pickle | cited: none
- g013 (single): How do I get the list of member names stored in a ZipFile archive? | failures: invalid citations | retrieved sources: zipfile | cited: none
- g014 (single): Which components does urllib.parse.urlparse split a URL into? | failures: invalid citations | retrieved sources: urllib.parse | cited: none
- g015 (single): Which re flag makes a regular expression match letters regardless of case? | failures: invalid citations | retrieved sources: re | cited: none
- g016 (single): How do I duplicate a nested list so that changing the inner lists of the duplicate leaves the original untouched? | failures: recall 0, invalid citations | retrieved sources: itertools, collections, random | cited: none
- g017 (single): How can I break a long list into consecutive groups of three items, where the last group may be smaller? | failures: invalid citations | retrieved sources: itertools, heapq, textwrap | cited: none
- g018 (single): How can I make my program produce the same sequence of random numbers on every run? | failures: invalid citations | retrieved sources: random | cited: none
- g019 (single): My file search with a '**' wildcard only looks one folder deep instead of descending into every subfolder; what am I missing? | failures: invalid citations | retrieved sources: glob, bisect, re | cited: none
- g020 (single): How do I get a scratch folder that is deleted automatically when I am finished with it? | failures: invalid citations | retrieved sources: tempfile, pickle | cited: none
- g021 (single): How do I compute a SHA-256 checksum of some bytes in Python? | failures: invalid citations | retrieved sources: hashlib | cited: none
- g022 (single): How do I turn a text value like '2024-03-15' into a date object? | failures: invalid citations | retrieved sources: datetime, calendar | cited: none
- g023 (single): How do I test whether two floating-point numbers are nearly equal rather than exactly equal? | failures: invalid citations | retrieved sources: decimal, math | cited: none
- g024 (single): How should I pass values typed in by a user into an SQL statement so they cannot inject SQL? | failures: invalid citations | retrieved sources: sqlite3 | cited: none
- g025 (single): How do I send my program's log messages to a file instead of the terminal? | failures: invalid citations | retrieved sources: logging, shutil | cited: none
- g026 (single): How do I run an external program and get back the text it wrote to standard output and standard error? | failures: invalid citations | retrieved sources: subprocess | cited: none
- g027 (single): How do I stop two threads from modifying the same shared variable at the same moment? | failures: invalid citations | retrieved sources: threading, sqlite3 | cited: none
- g028 (single): How do I make one of my unit tests be skipped only when some condition holds? | failures: invalid citations | retrieved sources: unittest, threading | cited: none
- g029 (single): How should I write a type hint for a parameter that may be an int or may be None? | failures: invalid citations | retrieved sources: typing | cited: none
- g030 (single): Is there a tidy way to ignore one particular exception raised inside a with block without writing try/except? | failures: invalid citations | retrieved sources: contextlib, logging | cited: none
- g031 (single): When I write rows of comma-separated data to a file on Windows, every row ends with an extra carriage return. What causes this? | failures: invalid citations | retrieved sources: csv | cited: none
- g032 (single): My recursive function recomputes the same results over and over and is very slow; is there a built-in way to remember the results of earlier calls? | failures: invalid citations | retrieved sources: heapq, threading, typing | cited: none
- g033 (single): After I serialise a Python tuple to JSON text and read it back, it comes back as a list. Is that expected? | failures: refused answerable question | retrieved sources: json, pickle | cited: none
- g034 (single): When I split the file name '.bashrc' into a base name and an extension, why is the extension empty? | failures: invalid citations | retrieved sources: os.path, shutil | cited: none
- g035 (single): Why do I get a TypeError when I check whether a timestamp that carries a time zone is earlier than one that does not? | failures: invalid citations | retrieved sources: datetime | cited: none
- g036 (single): The help text of my command-line tool shows the value placeholder of an option in all capital letters; how can I change what is shown there? | failures: invalid citations | retrieved sources: argparse | cited: none
- g037 (single): Why does my class decorated with @dataclass raise ValueError when I give a field an empty list as its default? | failures: invalid citations | retrieved sources: dataclasses, typing | cited: none
- g038 (single): My script hangs forever when the child program it started prints a lot of output and I wait for the child to exit; what is going on? | failures: invalid citations | retrieved sources: subprocess | cited: none
- g039 (single): How can I let my program exit even though a background worker is still looping forever? | failures: recall 0, invalid citations | retrieved sources: contextlib, subprocess | cited: none
- g040 (single): Is it safe to load saved Python objects from a file I downloaded from an unknown website? | failures: refused answerable question | retrieved sources: pickle, hashlib, unittest | cited: none
- g041 (multi): How do I turn an instance of a dataclass into a JSON string? | failures: invalid citations | retrieved sources: dataclasses, json | cited: none
- g042 (multi): How can I load the rows of a comma-separated text file into an SQLite table? | failures: invalid citations | retrieved sources: sqlite3 | cited: none
- g043 (multi): How do I copy an entire directory tree into a temporary directory that is removed automatically afterwards? | failures: invalid citations | retrieved sources: shutil, tempfile | cited: none
- g044 (multi): How do I move a date forward by one month, falling back to the last day of the next month when the same day number does not exist there (for example from January 31)? | failures: invalid citations | retrieved sources: datetime, calendar | cited: none
- g045 (multi): How do I combine all the integers in a list into one value with bitwise OR, without writing a lambda? | failures: recall 0, invalid citations | retrieved sources: itertools, random, heapq | cited: none
- g046 (multi): How do I count how often each pair of adjacent elements occurs in a sequence? | failures: invalid citations | retrieved sources: itertools, collections | cited: none
- g047 (multi): How do I put every .log file from a folder into a new zip archive? | failures: invalid citations | retrieved sources: zipfile, shutil | cited: none
- g048 (multi): How do I turn the query string of a URL into a dictionary and output that dictionary as JSON text? | failures: invalid citations | retrieved sources: json, urllib.parse, secrets | cited: none
- g049 (multi): How do I pick a random member of an Enum class? | failures: invalid citations | retrieved sources: enum | cited: none
- g050 (multi): How do I get the three tuples with the highest value in their second position from a large list, without sorting the whole list and without writing a lambda for the key? | failures: invalid citations | retrieved sources: itertools, heapq, functools | cited: none
- g051 (unanswerable): How do I run several asyncio coroutines concurrently and wait until all of them have finished? | failures: answered unanswerable question | retrieved sources: subprocess, threading, contextlib | cited: none
- g052 (unanswerable): How do I open a TCP connection to a server and send a block of bytes over it? | failures: answered unanswerable question | retrieved sources: threading, pickle, subprocess | cited: none
- g053 (unanswerable): How do I use a pool of worker processes to apply a function to every item of a list in parallel? | failures: answered unanswerable question | retrieved sources: threading, random, heapq | cited: none
- g054 (unanswerable): How do I parse an XML file and loop over all elements with a given tag name? | failures: answered unanswerable question | retrieved sources: argparse, urllib.parse, string | cited: none
- g055 (unanswerable): How do I start a simple web server that serves the files in the current folder over HTTP? | failures: answered unanswerable question | retrieved sources: pickle, os.path, contextlib | cited: none
- g056 (unanswerable): How do I generate a time-ordered version 7 UUID with the uuid module? | failures: answered unanswerable question | retrieved sources: uuid | cited: none
- g057 (unanswerable): Which hashlib function produces bcrypt password hashes? | failures: answered unanswerable question | retrieved sources: hashlib | cited: none
- g058 (unanswerable): How can a pathlib Path notify me when a file in its directory is created or modified? | failures: answered unanswerable question | retrieved sources: pathlib, os.path | cited: none
- g059 (unanswerable): How do I read a CSV file into a pandas DataFrame and group the rows by a column? | failures: answered unanswerable question | retrieved sources: csv | cited: none
- g060 (unanswerable): How do I declare a ForeignKey between two Django models? | failures: answered unanswerable question | retrieved sources: collections, functools, typing | cited: none
