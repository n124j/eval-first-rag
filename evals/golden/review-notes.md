# Golden set review notes

Human review pack for `golden.jsonl`. This file is not loaded by any code.

## About the set

- 60 items: 40 `single` (one expected source page), 10 `multi` (needs facts from two
  different pages; exactly two expected sources), 10 `unanswerable` (no expected sources;
  the system should decline).
- Answerable items cover 38 of the 41 pages (not covered: fractions, pathlib, time); no page is the expected source of more than
  three `single` items.
- Difficulty mix of the `single` items: 15 easy (the question reuses the page's own terms,
  usually the API name), 15 medium (paraphrased for a user who does not know the API name),
  10 hard (the question describes a symptom or behaviour and shares few keywords with the
  supporting passage).
- `multi` items were checked so that neither page alone mentions the other half of the
  answer (for example `dataclasses` never mentions JSON and `json` never mentions
  dataclasses).
- `unanswerable` items: 5 near-misses about real standard-library modules that are not in
  the 41 pages (asyncio, socket, multiprocessing, xml, http.server), 3 about features the
  included pages do not document (uuid version 7, bcrypt in hashlib, file watching in
  pathlib), and 2 plainly out of domain (pandas, Django). For each, `absent_terms.json`
  lists the terms an answer would need; an acceptance test asserts none of them occurs in
  any `corpus/*.txt` (case-insensitive).
- How it was written: by the tester role, by reading the plain-text pages in `corpus/`; no
  model or network was used. Each answerable item below carries a supporting excerpt copied
  from the named page. Excerpts were checked mechanically against the page text; line wraps
  in the source are shown here collapsed to single spaces.

## Items

### g001 - single (easy)

**Question:** When a value is already in a sorted list, where does bisect_left place the insertion point relative to the existing equal entries?

**Expected source(s):** `bisect`

Supporting excerpt from `bisect`:

```text
If *x* is already present in *a*, the insertion point will be before (to the left of) any existing entries.
```

### g002 - single (easy)

**Question:** In a heap managed by heapq, which element is always stored at index 0?

**Expected source(s):** `heapq`

Supporting excerpt from `heapq`:

```text
The interesting property of a heap is that its smallest element is always the root, "heap[0]".
```

### g003 - single (easy)

**Question:** What kind of string does secrets.token_hex return?

**Expected source(s):** `secrets`

Supporting excerpt from `secrets`:

```text
Return a random text string, in hexadecimal. The string has *nbytes* random bytes, each byte converted to two hex digits.
```

### g004 - single (easy)

**Question:** How does shutil.copy2 differ from shutil.copy?

**Expected source(s):** `shutil`

Supporting excerpt from `shutil`:

```text
Identical to "copy()" except that "copy2()" also attempts to preserve file metadata.
```

### g005 - single (easy)

**Question:** What is the difference between statistics.stdev and statistics.pstdev?

**Expected source(s):** `statistics`

Supporting excerpt from `statistics`:

```text
Return the population standard deviation (the square root of the population variance).
```

### g006 - single (easy)

**Question:** What does textwrap.dedent do to a block of text?

**Expected source(s):** `textwrap`

Supporting excerpt from `textwrap`:

```text
Remove any common leading whitespace from every line in *text*.
```

### g007 - single (easy)

**Question:** What does functools.partial return when I give it a function and some arguments?

**Expected source(s):** `functools`

Supporting excerpt from `functools`:

```text
Return a new partial object which when called will behave like *func* called with the positional arguments *args* and keyword arguments *keywords*.
```

### g008 - single (easy)

**Question:** What happens when I assign to a field of a dataclass declared with frozen=True?

**Expected source(s):** `dataclasses`

Supporting excerpt from `dataclasses`:

```text
*frozen*: If true (the default is "False"), assigning to fields will generate an exception.
```

### g009 - single (easy)

**Question:** What is the Decimal quantize method used for?

**Expected source(s):** `decimal`

Supporting excerpt from `decimal`:

```text
The "quantize()" method rounds a number to a fixed exponent. This method is useful for monetary applications that often round results to a fixed number of places:
```

### g010 - single (easy)

**Question:** Which function in the uuid module generates a random UUID?

**Expected source(s):** `uuid`

Supporting excerpt from `uuid`:

```text
uuid.uuid4() Generate a random UUID.
```

### g011 - single (easy)

**Question:** Which characters are included in string.punctuation?

**Expected source(s):** `string`

Supporting excerpt from `string`:

```text
String of ASCII characters which are considered punctuation characters in the "C" locale:
```

### g012 - single (easy)

**Question:** What does operator.itemgetter return when it is created with several items?

**Expected source(s):** `operator`

Supporting excerpt from `operator`:

```text
If multiple items are specified, returns a tuple of lookup values.
```

### g013 - single (easy)

**Question:** How do I get the list of member names stored in a ZipFile archive?

**Expected source(s):** `zipfile`

Supporting excerpt from `zipfile`:

```text
ZipFile.namelist() Return a list of archive members by name.
```

### g014 - single (easy)

**Question:** Which components does urllib.parse.urlparse split a URL into?

**Expected source(s):** `urllib.parse`

Supporting excerpt from `urllib.parse`:

```text
Parse a URL into six components, returning a 6-item *named tuple*. This corresponds to the general structure of a URL: "scheme://netloc/path;parameters?query#fragment".
```

### g015 - single (easy)

**Question:** Which re flag makes a regular expression match letters regardless of case?

**Expected source(s):** `re`

Supporting excerpt from `re`:

```text
Perform case-insensitive matching; expressions like "[A-Z]" will also match lowercase letters.
```

### g016 - single (medium)

**Question:** How do I duplicate a nested list so that changing the inner lists of the duplicate leaves the original untouched?

**Expected source(s):** `copy`

Supporting excerpt from `copy`:

```text
A *deep copy* constructs a new compound object and then, recursively, inserts *copies* into it of the objects found in the original.
```

### g017 - single (medium)

**Question:** How can I break a long list into consecutive groups of three items, where the last group may be smaller?

**Expected source(s):** `itertools`

Supporting excerpt from `itertools`:

```text
Batch data from the *iterable* into tuples of length *n*. The last batch may be shorter than *n*.
```

### g018 - single (medium)

**Question:** How can I make my program produce the same sequence of random numbers on every run?

**Expected source(s):** `random`

Supporting excerpt from `random`:

```text
Initialize the random number generator.
```

### g019 - single (medium)

**Question:** My file search with a '**' wildcard only looks one folder deep instead of descending into every subfolder; what am I missing?

**Expected source(s):** `glob`

Supporting excerpt from `glob`:

```text
If *recursive* is true, the pattern ""**"" will match any files and zero or more directories, subdirectories and symbolic links to directories.
```

### g020 - single (medium)

**Question:** How do I get a scratch folder that is deleted automatically when I am finished with it?

**Expected source(s):** `tempfile`

Supporting excerpt from `tempfile`:

```text
This class securely creates a temporary directory using the same rules as "mkdtemp()". The resulting object can be used as a *context manager*
```

### g021 - single (medium)

**Question:** How do I compute a SHA-256 checksum of some bytes in Python?

**Expected source(s):** `hashlib`

Supporting excerpt from `hashlib`:

```text
"sha256()" to create a SHA-256 hash object. You can now feed this object with *bytes-like objects* (normally "bytes") using the "update" method.
```

### g022 - single (medium)

**Question:** How do I turn a text value like '2024-03-15' into a date object?

**Expected source(s):** `datetime`

Supporting excerpt from `datetime`:

```text
date.fromisoformat('2019-12-04')
```

### g023 - single (medium)

**Question:** How do I test whether two floating-point numbers are nearly equal rather than exactly equal?

**Expected source(s):** `math`

Supporting excerpt from `math`:

```text
Return "True" if the values *a* and *b* are close to each other and "False" otherwise.
```

### g024 - single (medium)

**Question:** How should I pass values typed in by a user into an SQL statement so they cannot inject SQL?

**Expected source(s):** `sqlite3`

Supporting excerpt from `sqlite3`:

```text
Always use placeholders instead of string formatting to bind Python values to SQL statements, to avoid SQL injection attacks
```

### g025 - single (medium)

**Question:** How do I send my program's log messages to a file instead of the terminal?

**Expected source(s):** `logging`

Supporting excerpt from `logging`:

```text
logging.basicConfig(filename='myapp.log', level=logging.INFO)
```

### g026 - single (medium)

**Question:** How do I run an external program and get back the text it wrote to standard output and standard error?

**Expected source(s):** `subprocess`

Supporting excerpt from `subprocess`:

```text
If *capture_output* is true, stdout and stderr will be captured.
```

### g027 - single (medium)

**Question:** How do I stop two threads from modifying the same shared variable at the same moment?

**Expected source(s):** `threading`

Supporting excerpt from `threading`:

```text
Once a thread has acquired a lock, subsequent attempts to acquire it block, until it is released; any thread may release it.
```

### g028 - single (medium)

**Question:** How do I make one of my unit tests be skipped only when some condition holds?

**Expected source(s):** `unittest`

Supporting excerpt from `unittest`:

```text
Skip the decorated test if *condition* is true.
```

### g029 - single (medium)

**Question:** How should I write a type hint for a parameter that may be an int or may be None?

**Expected source(s):** `typing`

Supporting excerpt from `typing`:

```text
"Optional[X]" is equivalent to "X | None" (or "Union[X, None]").
```

### g030 - single (medium)

**Question:** Is there a tidy way to ignore one particular exception raised inside a with block without writing try/except?

**Expected source(s):** `contextlib`

Supporting excerpt from `contextlib`:

```text
Return a context manager that suppresses any of the specified exceptions if they occur in the body of a "with" statement
```

### g031 - single (hard)

**Question:** When I write rows of comma-separated data to a file on Windows, every row ends with an extra carriage return. What causes this?

**Expected source(s):** `csv`

Supporting excerpt from `csv`:

```text
on platforms that use "\r\n" linendings on write an extra "\r" will be added.
```

### g032 - single (hard)

**Question:** My recursive function recomputes the same results over and over and is very slow; is there a built-in way to remember the results of earlier calls?

**Expected source(s):** `functools`

Supporting excerpt from `functools`:

```text
Simple lightweight unbounded function cache. Sometimes called "memoize".
```

### g033 - single (hard)

**Question:** After I serialise a Python tuple to JSON text and read it back, it comes back as a list. Is that expected?

**Expected source(s):** `json`

Supporting excerpt from `json`:

```text
| list, tuple | array |
```

### g034 - single (hard)

**Question:** When I split the file name '.bashrc' into a base name and an extension, why is the extension empty?

**Expected source(s):** `os.path`

Supporting excerpt from `os.path`:

```text
Leading periods of the last component of the path are considered to be part of the root:
```

### g035 - single (hard)

**Question:** Why do I get a TypeError when I check whether a timestamp that carries a time zone is earlier than one that does not?

**Expected source(s):** `datetime`

Supporting excerpt from `datetime`:

```text
Order comparison between naive and aware "datetime" objects, as well as a "datetime" object and a "date" object that is not also a "datetime" instance, raises "TypeError".
```

### g036 - single (hard)

**Question:** The help text of my command-line tool shows the value placeholder of an option in all capital letters; how can I change what is shown there?

**Expected source(s):** `argparse`

Supporting excerpt from `argparse`:

```text
By default, for positional argument actions, the dest value is used directly, and for optional argument actions, the dest value is uppercased.
```

### g037 - single (hard)

**Question:** Why does my class decorated with @dataclass raise ValueError when I give a field an empty list as its default?

**Expected source(s):** `dataclasses`

Supporting excerpt from `dataclasses`:

```text
x: list = [] # This code raises ValueError
```

### g038 - single (hard)

**Question:** My script hangs forever when the child program it started prints a lot of output and I wait for the child to exit; what is going on?

**Expected source(s):** `subprocess`

Supporting excerpt from `subprocess`:

```text
This will deadlock when using "stdout=PIPE" or "stderr=PIPE" and the child process generates enough output to a pipe such that it blocks waiting for the OS pipe buffer to accept more data.
```

### g039 - single (hard)

**Question:** How can I let my program exit even though a background worker is still looping forever?

**Expected source(s):** `threading`

Supporting excerpt from `threading`:

```text
The significance of this flag is that the entire Python program exits when only daemon threads are left.
```

### g040 - single (hard)

**Question:** Is it safe to load saved Python objects from a file I downloaded from an unknown website?

**Expected source(s):** `pickle`

Supporting excerpt from `pickle`:

```text
The "pickle" module **is not secure**. Only unpickle data you trust.
```

### g041 - multi

**Question:** How do I turn an instance of a dataclass into a JSON string?

**Expected source(s):** `dataclasses`, `json`

Supporting excerpt from `dataclasses`:

```text
Converts the dataclass *obj* to a dict (by using the factory function *dict_factory*).
```

Supporting excerpt from `json`:

```text
Serialize *obj* to a JSON formatted "str" using this conversion table.
```

### g042 - multi

**Question:** How can I load the rows of a comma-separated text file into an SQLite table?

**Expected source(s):** `csv`, `sqlite3`

Supporting excerpt from `csv`:

```text
Return a reader object that will process lines from the given *csvfile*.
```

Supporting excerpt from `sqlite3`:

```text
cur.executemany("INSERT INTO movie VALUES(?, ?, ?)", data)
```

### g043 - multi

**Question:** How do I copy an entire directory tree into a temporary directory that is removed automatically afterwards?

**Expected source(s):** `tempfile`, `shutil`

Supporting excerpt from `tempfile`:

```text
This class securely creates a temporary directory using the same rules as "mkdtemp()".
```

Supporting excerpt from `shutil`:

```text
Recursively copy an entire directory tree rooted at *src* to a directory named *dst* and return the destination directory.
```

### g044 - multi

**Question:** How do I move a date forward by one month, falling back to the last day of the next month when the same day number does not exist there (for example from January 31)?

**Expected source(s):** `calendar`, `datetime`

Supporting excerpt from `calendar`:

```text
Returns weekday of first day of the month and number of days in month, for the specified *year* and *month*.
```

Supporting excerpt from `datetime`:

```text
Return a new "date" object with the same values, but with specified parameters updated.
```

### g045 - multi

**Question:** How do I combine all the integers in a list into one value with bitwise OR, without writing a lambda?

**Expected source(s):** `functools`, `operator`

Supporting excerpt from `functools`:

```text
Apply *function* of two arguments cumulatively to the items of *iterable*, from left to right, so as to reduce the iterable to a single value.
```

Supporting excerpt from `operator`:

```text
Return the bitwise or of *a* and *b*.
```

### g046 - multi

**Question:** How do I count how often each pair of adjacent elements occurs in a sequence?

**Expected source(s):** `itertools`, `collections`

Supporting excerpt from `itertools`:

```text
Return successive overlapping pairs taken from the input *iterable*.
```

Supporting excerpt from `collections`:

```text
A "Counter" is a "dict" subclass for counting *hashable* objects.
```

### g047 - multi

**Question:** How do I put every .log file from a folder into a new zip archive?

**Expected source(s):** `glob`, `zipfile`

Supporting excerpt from `glob`:

```text
Return a possibly empty list of path names that match *pathname*,
```

Supporting excerpt from `zipfile`:

```text
Write the file named *filename* to the archive, giving it the archive name *arcname*
```

### g048 - multi

**Question:** How do I turn the query string of a URL into a dictionary and output that dictionary as JSON text?

**Expected source(s):** `urllib.parse`, `json`

Supporting excerpt from `urllib.parse`:

```text
Parse a query string given as a string argument (data of type *application/x-www-form-urlencoded*). Data are returned as a dictionary.
```

Supporting excerpt from `json`:

```text
Serialize *obj* to a JSON formatted "str" using this conversion table.
```

### g049 - multi

**Question:** How do I pick a random member of an Enum class?

**Expected source(s):** `enum`, `random`

Supporting excerpt from `enum`:

```text
can be iterated over to return its canonical (i.e. non-alias) members in definition order
```

Supporting excerpt from `random`:

```text
Return a random element from the non-empty sequence *seq*.
```

### g050 - multi

**Question:** How do I get the three tuples with the highest value in their second position from a large list, without sorting the whole list and without writing a lambda for the key?

**Expected source(s):** `heapq`, `operator`

Supporting excerpt from `heapq`:

```text
Return a list with the *n* largest elements from the dataset defined by *iterable*. *key*, if provided, specifies a function of one argument
```

Supporting excerpt from `operator`:

```text
After "f = itemgetter(2)", the call "f(r)" returns "r[2]".
```

### g051 - unanswerable (near-miss: module not included (asyncio))

**Question:** How do I run several asyncio coroutines concurrently and wait until all of them have finished?

**Expected source(s):** none

**Why unanswerable:** asyncio has no page of its own; several included pages mention asyncio in passing (contextlib, subprocess, threading, unittest), but none explains running coroutines concurrently.

**Absent terms (none occur in any corpus/*.txt, case-insensitive):** `asyncio.gather`, `taskgroup`, `create_task`, `asyncio.run(`

### g052 - unanswerable (near-miss: module not included (socket))

**Question:** How do I open a TCP connection to a server and send a block of bytes over it?

**Expected source(s):** none

**Why unanswerable:** The socket module is not included; 'socket' and 'tcp' occur only in passing (e.g. logging handlers, pickling notes), never with connection or send APIs.

**Absent terms (none occur in any corpus/*.txt, case-insensitive):** `socket.socket(`, `create_connection`, `sendall`, `af_inet`

### g053 - unanswerable (near-miss: module not included (multiprocessing))

**Question:** How do I use a pool of worker processes to apply a function to every item of a list in parallel?

**Expected source(s):** none

**Why unanswerable:** multiprocessing is not included; the threading page only advises using multiprocessing or ProcessPoolExecutor for CPU-bound work and does not explain process pools.

**Absent terms (none occur in any corpus/*.txt, case-insensitive):** `multiprocessing.pool`, `pool.map`, `pool(processes`, `worker process`

### g054 - unanswerable (near-miss: module not included (xml))

**Question:** How do I parse an XML file and loop over all elements with a given tag name?

**Expected source(s):** none

**Why unanswerable:** No XML module is included; 'xml' occurs only once, in the pickle page's comparison of serialisation formats.

**Absent terms (none occur in any corpus/*.txt, case-insensitive):** `xml.etree`, `elementtree`, `iterparse`

### g055 - unanswerable (near-miss: module not included (http.server))

**Question:** How do I start a simple web server that serves the files in the current folder over HTTP?

**Expected source(s):** none

**Why unanswerable:** http.server is not included; urllib.parse talks about HTTP URLs, which a keyword match could confuse with serving HTTP.

**Absent terms (none occur in any corpus/*.txt, case-insensitive):** `http.server`, `simplehttprequesthandler`, `httpserver`, `serve_forever`

### g056 - unanswerable (included module, undocumented feature (uuid))

**Question:** How do I generate a time-ordered version 7 UUID with the uuid module?

**Expected source(s):** none

**Why unanswerable:** The 3.12 uuid page documents uuid1, uuid3, uuid4 and uuid5 only; version 7 UUIDs are not described.

**Absent terms (none occur in any corpus/*.txt, case-insensitive):** `uuid7`, `version 7`, `time-ordered`

### g057 - unanswerable (included module, undocumented feature (hashlib))

**Question:** Which hashlib function produces bcrypt password hashes?

**Expected source(s):** none

**Why unanswerable:** The hashlib page documents pbkdf2_hmac and scrypt for passwords; bcrypt is never mentioned.

**Absent terms (none occur in any corpus/*.txt, case-insensitive):** `bcrypt`, `hashlib.bcrypt`, `$2b$`

### g058 - unanswerable (included module, undocumented feature (pathlib))

**Question:** How can a pathlib Path notify me when a file in its directory is created or modified?

**Expected source(s):** none

**Why unanswerable:** pathlib has no file-watching facility and the page does not describe change notification.

**Absent terms (none occur in any corpus/*.txt, case-insensitive):** `inotify`, `watchdog`, `file system events`, `change notification`, `directory changes`

### g059 - unanswerable (out of domain (pandas))

**Question:** How do I read a CSV file into a pandas DataFrame and group the rows by a column?

**Expected source(s):** none

**Why unanswerable:** pandas is a third-party library; the csv page could mislead a keyword retriever but says nothing about DataFrames.

**Absent terms (none occur in any corpus/*.txt, case-insensitive):** `pandas`, `dataframe`, `read_csv`, `pd.`

### g060 - unanswerable (out of domain (Django))

**Question:** How do I declare a ForeignKey between two Django models?

**Expected source(s):** none

**Why unanswerable:** Django is a third-party web framework; it is only named once in collections (ChainMap example), with nothing about models.

**Absent terms (none occur in any corpus/*.txt, case-insensitive):** `foreignkey`, `models.model`, `on_delete`

## Known limitations of this set

- It is small: 60 items. Metric differences of a few points between runs can be noise; one
  item is worth 2.5 points of recall on the 40 `single` items.
- It was written by the workforce (the tester role) from the pages themselves, so question
  wording is biased towards the vocabulary of the docs, even for the medium and hard items.
- Scoring is at document level: a retrieved chunk counts as relevant if it comes from an
  expected doc, even if it is not the chunk that holds the answer.
- It is single-author: one writer drafted every question and excerpt, with no independent
  second annotation. The director's review is the only cross-check before the baseline.
- Some `multi` items have plausible alternative second sources outside the expected pair
  (for example `pathlib` instead of `glob` for listing files); document-level scoring will
  count those as misses.
- Single items were screened for obvious second answering pages (two candidates that
  pathlib also answers were replaced), but the screen was by keyword, not exhaustive; a few
  APIs are mentioned in passing on other pages (for example `deepcopy` in `dataclasses`,
  `basicConfig` in `contextlib`).
- The "undocumented feature" unanswerable items (uuid version 7, bcrypt) can be answered
  negatively by inference ("the page lists the supported functions and this is not one of
  them"); a system that says so with a citation is arguably right rather than wrong.
