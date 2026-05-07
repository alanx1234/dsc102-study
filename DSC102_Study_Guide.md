# DSC 102 Study Guide

This guide is built from the lecture PDFs and practice/midterm-review questions in this repo. The repeated question patterns are the best signal for what to master: performance arithmetic, memory hierarchy, data representation, OS scheduling, paging, storage/I/O, data layout, and parallelism.

## How to Use This

Read each concept section first, then do the "problem templates" without looking at the answer. For this class, knowing the vocabulary is not enough. You need to be able to turn a systems story into arithmetic:

1. Identify the resource: CPU cycles, cache accesses, pages, disk blocks, workers, or bytes.
2. Write the governing formula.
3. Convert units carefully.
4. Simulate state changes when needed.
5. Interpret the result in systems language.

Common unit anchors:

- 1 byte = 8 bits.
- k bits represent `2^k` distinct bit patterns.
- To represent `N` unique items, need `ceil(log2(N))` bits.
- For byte-addressable memory, one address names one byte.
- 1 KB is often `2^10 = 1024` bytes in memory/page problems.
- Practice questions sometimes say `1 GB = 10^9` bytes; follow the problem statement.
- Clock rate in Hz means cycles per second.
- Runtime = cycles / clock rate.

## 1. Big Picture: Systems for Scalable Analytics

The course is about why data science performance depends on the whole system, not just the algorithm's math. A model can be mathematically elegant and still be slow because data is stuck on disk, crossing the network, missing cache, waiting on memory, or blocked by serial work.

The main systems question is:

> Where is the bottleneck, and what abstraction is hiding it?

Important performance measures:

- Latency: time to finish one operation or one job.
- Throughput: amount of work completed per unit time.
- Scalability: ability to maintain or improve useful performance as data, users, or hardware resources grow.
- Speedup: `T_1 / T_p`, where `T_1` is time with one worker and `T_p` is time with `p` workers.
- Efficiency: `speedup / p`.

The central tradeoff in scalable analytics:

- More data can improve accuracy and enable new applications.
- More data creates pressure on memory, storage, CPU, networking, scheduling, and parallel coordination.
- Systems design is about controlling these costs.

## 2. Computer Organization

### Hardware and Software Layers

Hardware is the physical machinery: CPU, GPU, DRAM, caches, buses, disk, network devices.

Software is layered:

- Application code: Python, data science scripts, ML training code.
- Runtime/platform: Python interpreter, JVM, Spark, Dask, TensorFlow, PyTorch.
- Operating system: Linux, Windows, macOS.
- Firmware/hardware control.
- Machine instructions executed by the processor.

Abstraction means a higher layer uses an interface without needing every lower-level implementation detail. But scalable analytics often requires crossing abstraction layers because performance bugs live below the level where the code was written.

### Stored Program Model

In a Von Neumann machine:

- Instructions and data are both stored as bit sequences.
- A program counter/instruction pointer tells the CPU where the next instruction is.
- The processor fetches, decodes, and executes instructions mostly in program order, except when branches, jumps, or other control-flow instructions change the order.

Dataflow is a different model: operations run when their inputs are available. This idea shows up later in dataflow graphs, task graphs, Dask, Spark, and ML computation graphs.

### ISA and Processor Commands

An Instruction Set Architecture, or ISA, is the vocabulary of machine commands a CPU understands.

High-yield ISA command categories:

- Memory access: load from memory to register, store from register to memory.
- Arithmetic/logic: add, multiply, compare, boolean operations.
- Control flow: branch, jump, call, return.

Registers are tiny, very fast storage inside the CPU. Caches are larger but slower than registers. DRAM is much larger but much slower than cache. Disk/SSD is persistent but far slower than DRAM.

### CPU Performance Arithmetic

Core formulas:

```text
average CPI = sum(instruction_fraction_i * cycles_i)
total cycles = instruction count * average CPI
execution time = total cycles / clock rate
```

If memory instructions include stalls:

```text
base cycles = instruction count * base CPI
stall cycles = memory_instruction_count * stall_cycles_per_memory_instruction
total cycles = base cycles + stall cycles
```

Speedup after an optimization:

```text
speedup = old time / new time = old cycles / new cycles
```

Worked example:

A processor runs at 3 GHz. A program has 2 billion instructions:

- 40% arithmetic at 1 cycle.
- 30% memory at 5 cycles.
- 30% control at 2 cycles.

Average CPI:

```text
CPI = 0.4*1 + 0.3*5 + 0.3*2 = 2.5
cycles = 2e9 * 2.5 = 5e9
time = 5e9 / 3e9 = 1.67 seconds
```

If memory instructions become 50%, arithmetic 20%, control 30%:

```text
new CPI = 0.2*1 + 0.5*5 + 0.3*2 = 3.3
new time = 2e9 * 3.3 / 3e9 = 2.2 seconds
```

Why this matters: memory-heavy workloads can dominate runtime even when the CPU clock is fast.

## 3. Digital Representation

### Bits, Bytes, and Capacity

A bit is 0 or 1. A byte is 8 bits.

With `k` bits:

```text
number of distinct patterns = 2^k
```

To represent `N` unique values:

```text
minimum bits = ceil(log2(N))
```

Examples:

- 5 bits represent `2^5 = 32` values.
- 3 bytes = 24 bits, so `2^24 = 16,777,216` values.
- 70,000 unique items need 17 bits because `2^16 = 65,536 < 70,000 < 2^17 = 131,072`.

### Binary, Decimal, Hex, and Octal

Hexadecimal is base 16 and maps neatly to binary because each hex digit is 4 bits:

```text
0 = 0000
1 = 0001
2 = 0010
...
A = 1010
B = 1011
C = 1100
D = 1101
E = 1110
F = 1111
```

Example:

```text
0x7F3A = 0111 1111 0011 1010
```

A 4-hex-digit address has 16 bits, which is 2 bytes. If each address points to 1 byte, then the address space contains:

```text
2^16 bytes = 65,536 bytes = 64 KiB
```

For mixed-base arithmetic:

```text
1011_2 = 11
1A_16 = 26
12_8 = 10
1011_2 + 1A_16 - 12_8 = 11 + 26 - 10 = 27
```

### Address Alignment

If each instruction is `b` bytes and instructions must be `b`-byte aligned, valid instruction addresses are multiples of `b`.

Examples:

- 4-byte aligned addresses have last 2 bits equal to `00`; valid fraction is `1/4`.
- 8-byte aligned addresses have last 3 bits equal to `000`; valid fraction is `1/8`.

Example:

Each instruction is 8 bytes, PC starts at `0x4000`.

```text
0x4000, 0x4008, 0x4010, 0x4018, 0x4020
```

If instruction memory is 2 KB:

```text
2 KB = 2048 bytes
instructions = 2048 / 8 = 256
```

### Unsigned and Signed Integers

Unsigned `k`-bit integers range from:

```text
0 to 2^k - 1
```

Signed two's-complement `k`-bit integers range from:

```text
-2^(k-1) to 2^(k-1) - 1
```

For 8-bit signed integers:

```text
min = -128
max = 127
```

Two's-complement interpretation:

- If the most significant bit is 0, value is nonnegative.
- If the most significant bit is 1, value is negative.
- To get the negative value, either:
  - compute unsigned value minus `2^k`, or
  - invert bits, add 1, and attach a minus sign.

Examples:

```text
10000010 as unsigned = 130
as signed 8-bit = 130 - 256 = -126
```

```text
10011011 as unsigned = 155
as signed 8-bit = 155 - 256 = -101
```

```text
10000111 as unsigned = 135
as signed 8-bit = 135 - 256 = -121
```

Overflow rule for signed addition:

- Adding two positives should produce positive.
- Adding two negatives should produce negative.
- If the sign flips, signed overflow occurred.

Example:

```text
120 + 10 = 130
130 is larger than 127, so 8-bit signed overflow occurs.
machine bit pattern = 10000010
interpreted signed value = -126
```

### Floating Point

Floating point stores numbers in binary scientific notation.

For IEEE-style formats, the idea is:

```text
value = (-1)^sign * 2^(exponent - bias) * mantissa
```

For normal IEEE float32:

- 1 sign bit.
- 8 exponent bits.
- 23 fraction/mantissa bits.
- Bias is 127.
- Exponent all 1s is reserved for infinity/NaN.

The custom 6-bit float in the practice questions uses:

- bit 5: sign.
- bits 4-3: exponent.
- bits 2-0: fraction.
- value = `(-1)^s * 2^(exp - 1) * (1 + f2*2^-1 + f1*2^-2 + f0*2^-3)`.

Example: `010110`

```text
s = 0, so positive
exp = 10_2 = 2, so scale = 2^(2 - 1) = 2
fraction = 110, so mantissa = 1 + 1/2 + 1/4 + 0/8 = 1.75
value = 2 * 1.75 = 3.5
```

Float warnings:

- Floating point arithmetic is approximate.
- Addition/multiplication are not always associative.
- Lower precision, such as float16, can improve speed and memory use but can harm accuracy.
- Quantization trades precision for smaller/faster models.

### Characters, Endianness, and Serialization

Characters are represented by encodings such as ASCII and Unicode. Strings are sequences of encoded characters.

Endianness is the byte order used when storing multi-byte values:

- Big-endian stores the most significant byte first.
- Little-endian stores the least significant byte first.

Serialization converts an in-memory data structure into bytes for storage or transmission. Deserialization reconstructs the object from bytes.

## 4. Memory Hierarchy and Locality

### Hierarchy

From fastest/smallest to slowest/largest:

```text
registers -> L1/L2/L3 cache -> DRAM -> SSD/HDD -> network/remote storage
```

The reason this hierarchy matters is that the CPU can execute instructions far faster than data can always be supplied from memory or disk.

### Locality of Reference

Temporal locality:

- If data was accessed recently, it may be accessed again soon.
- Loops often reuse variables and instructions.

Spatial locality:

- If an address was accessed, nearby addresses may be accessed soon.
- Arrays and sequential instructions benefit strongly from this.

Caching exploits both:

- Bring a block/cache line from lower memory to higher memory.
- Use nearby/reused data before evicting it.

### AMAT

Average Memory Access Time is one of the most repeated formulas.

Two-level cache/DRAM version:

```text
AMAT = cache_hit_time + miss_rate * miss_penalty
miss_rate = 1 - hit_rate
```

If the problem phrases the lower level as full memory access time rather than miss penalty, many class problems still use:

```text
AMAT = cache_time + miss_rate * memory_time
```

Three-level version:

```text
AMAT = L1_time + L1_miss_rate * (L2_time + L2_miss_rate * DRAM_time)
```

Example:

L1 = 2 ns, L1 hit rate = 0.8. L2 = 12 ns, L2 hit rate given L1 miss = 0.75. DRAM = 100 ns.

```text
L1 miss rate = 0.2
L2 miss rate = 0.25
AMAT = 2 + 0.2 * (12 + 0.25 * 100)
     = 2 + 0.2 * 37
     = 9.4 ns
```

Fraction reaching DRAM in a three-level hierarchy:

```text
P(DRAM) = L1_miss_rate * L2_miss_rate
```

### Cache Lines and Array Layout

A cache line transfers a contiguous block of bytes.

If cache line size is 64 bytes:

- float64 is 8 bytes, so 8 float64 values per cache line.
- int32 is 4 bytes, so 16 int32 values per cache line.

For row-major matrices, elements in the same row are contiguous:

```text
Version A:
for i in rows:
    for j in cols:
        total += M[i][j]
```

This walks through memory contiguously and uses spatial locality.

```text
Version B:
for j in cols:
    for i in rows:
        total += M[i][j]
```

This jumps by a large stride through memory. It causes many more cache misses and can lead to cache thrashing.

Approximate miss fraction for simple sequential scan:

```text
misses ~= 1 per cache line
miss fraction ~= 1 / elements_per_cache_line
```

For 64-byte cache lines and float64 values:

```text
8 elements per line
miss fraction ~= 1/8
hit fraction ~= 7/8
```

### Why NumPy Beats Python Loops

Native Python loops do repeated interpretation, type checks, object handling, and per-element overhead. NumPy stores homogeneous typed arrays and calls optimized compiled kernels, often using vectorization, cache-friendly layouts, and BLAS-like libraries.

For data science, prefer:

- vectorized operations,
- contiguous arrays,
- optimized libraries,
- avoiding Python-level per-element loops when possible.

## 5. Operating Systems

### What an OS Does

An operating system:

- Virtualizes hardware resources.
- Manages contention for CPU, memory, storage, GPUs, and network.
- Provides system-call APIs.
- Provides abstractions: process, thread, virtual memory, file, directory.
- Enforces protection and isolation.

OS design vocabulary:

- Abstraction: what interface is exposed.
- Mechanism: how the OS can do something.
- Policy: which choice the OS makes among options.

### Processes

A process is a running program.

The OS creates a process by:

1. Assigning a process ID.
2. Creating an address space.
3. Loading code/static data.
4. Setting up stack/heap and arguments.
5. Marking it ready/runnable.

Common states:

- Running: currently on CPU.
- Ready/runnable: able to run, waiting for CPU.
- Blocked/waiting: waiting for I/O or another event.

Important transition:

- Blocked -> Ready when the awaited I/O/event completes.

Process-related calls:

- `fork()`: create child process.
- `wait()`: parent waits for child to finish.
- `exec()`: replace current process image with a new program.
- `kill()`: send signal/stop.

Fork output rule:

If code prints once before `fork()`, that print happens once. Code after `fork()` runs in both parent and child. If parent calls `wait()`, the parent's later output happens after the child exits. Exact interleaving can still vary unless the program synchronizes output.

### Direct Execution, System Calls, Context Switches

The OS wants user code to run directly on hardware for speed, but safely.

Mechanisms:

- User mode prevents arbitrary hardware control.
- Kernel mode allows privileged operations.
- System calls transfer control to the OS for protected services.
- Timer interrupts let the OS regain control.
- Context switch saves one process/thread state and loads another.

Context switches cost time, and some scheduling questions explicitly charge this overhead.

### Threads

A thread is a lightweight execution unit inside a process.

Processes:

- Separate address spaces.
- Stronger isolation.
- Communication requires IPC, files, sockets, shared memory, etc.

Threads:

- Share the same process address space.
- Cheaper to create/switch.
- Easier shared-memory communication.
- Bugs can corrupt shared state.

Python note:

- The Global Interpreter Lock (GIL) means only one thread executes Python bytecode at a time in standard CPython.
- Python threads can still help for I/O-bound work.
- CPU-bound Python often needs multiprocessing, native libraries, NumPy, Numba, C extensions, or distributed tools.

## 6. Scheduling

### Metrics

For each job/process:

```text
turnaround time = completion time - arrival time
response time = first run time - arrival time
waiting time = total time spent ready but not running
```

Average turnaround time is common for batch performance. Response time matters for interactive jobs.

### Algorithms

FCFS/FIFO:

- Run jobs in arrival order.
- Simple and fair by arrival.
- Can suffer convoy effect: a long job delays short jobs.

SJF, non-preemptive:

- When CPU becomes free, run shortest available job.
- Often improves average turnaround.
- Requires knowing/estimating job length.

SRTF/SRCTF, preemptive:

- Always run job with shortest remaining time.
- New short jobs can preempt current job.
- Usually strong for turnaround, less fair to long jobs.

Round Robin:

- Each runnable job gets a time quantum.
- If unfinished after quantum, it goes to back of queue.
- Good response/fairness.
- Too-small quantum creates high context-switch overhead.
- Too-large quantum behaves more like FCFS.

### Scheduling Simulation Recipe

Use this every time:

1. Draw a time axis.
2. Track arrivals.
3. Track remaining burst times.
4. At each decision point, choose according to policy.
5. Include context switch overhead if the problem says so.
6. Record first run time and completion time.
7. Compute turnaround/response/waiting averages.

Example with context switches:

All arrive at 0:

```text
P1 = 20 ms, P2 = 5 ms, P3 = 15 ms, P4 = 10 ms
context switch = 1 ms before each process except first
```

FIFO order P1, P2, P3, P4:

```text
P1 starts 0
P2 starts 20 + 1 = 21
P3 starts 21 + 5 + 1 = 27
P4 starts 27 + 15 + 1 = 43
average waiting = (0 + 21 + 27 + 43) / 4 = 22.75 ms
```

SJF order P2, P4, P3, P1:

```text
P2 starts 0
P4 starts 5 + 1 = 6
P3 starts 6 + 10 + 1 = 17
P1 starts 17 + 15 + 1 = 33
average waiting = (0 + 6 + 17 + 33) / 4 = 14 ms
```

Context switch cost:

```text
4 jobs means 3 switches, so 3 ms total
```

## 7. Virtual Memory and Paging

### Address Space

Each process sees a virtual address space. The OS and hardware translate virtual addresses to physical addresses.

Address space segments:

- Code/text: program instructions.
- Stack: function calls, local variables, return addresses.
- Heap: dynamically allocated objects/data structures.

Virtual memory gives:

- isolation between processes,
- simpler programming model,
- ability to use disk-backed swap,
- flexible memory allocation.

### Pages and Frames

A page is a fixed-size chunk of virtual memory.

A page frame is a fixed-size slot in physical memory.

Typical page sizes: 4 KB to 16 KB.

For address translation:

```text
virtual address = VPN + offset
physical address = PFN + offset
```

If page size is `2^k` bytes:

```text
offset bits = k
VPN bits = virtual_address_bits - k
number of virtual pages = 2^(VPN bits)
```

Example:

32-bit virtual address, 4 KB pages:

```text
4 KB = 4096 = 2^12
offset bits = 12
VPN bits = 32 - 12 = 20
flat page table entries = 2^20 = 1,048,576
```

### Address Translation Example

Assume:

- 8-bit virtual address.
- Page size = 16 bytes = `2^4`, so offset has 4 bits.
- VPN -> PFN mapping: `0->5`, `1->2`, `2->7`, `3->1`.

Translate virtual address 35:

```text
VPN = floor(35 / 16) = 2
offset = 35 mod 16 = 3
PFN = 7
physical address = 7*16 + 3 = 115
```

Translate virtual address 50:

```text
VPN = floor(50 / 16) = 3
offset = 50 mod 16 = 2
PFN = 1
physical address = 1*16 + 2 = 18
```

### Page Table Size

Recipe:

1. Compute number of virtual pages per process.
2. Compute bytes per page table entry.
3. Multiply by number of processes.
4. If asked for physical frames, divide by page size and round up.

Example:

24-bit virtual address, page size 4 KB, PTE = 4 bytes, 8 processes.

```text
virtual address space = 2^24 bytes
page size = 2^12 bytes
virtual pages per process = 2^(24 - 12) = 2^12 = 4096
one page table = 4096 * 4 = 16,384 bytes = 16 KB
all page tables = 8 * 16 KB = 128 KB
```

### Page Faults and Replacement

A page fault happens when a referenced page is not currently in DRAM. The OS must bring it from disk/swap or another lower level.

Policies:

- FIFO: evict the page loaded earliest.
- LRU: evict the page least recently used.
- OPT: evict the page whose next use is farthest in the future. Not implementable in real life, but useful as an ideal lower bound.

Simulation recipe:

1. Write frames as slots.
2. Process reference string left to right.
3. On hit: update recency for LRU.
4. On fault: if empty frame exists, fill it; otherwise evict by policy.
5. Count faults.

Thrashing:

- The system spends too much time paging instead of doing useful CPU work.
- CPU utilization decreases because processes wait on memory/disk.

Page size tradeoff:

- Larger pages can improve spatial locality and reduce page table entries.
- Larger pages can increase internal fragmentation.
- Paging generally avoids external fragmentation.

## 8. Files, Databases, Data Layout, and Storage

### Filesystem

A file is a persistent sequence of bytes with metadata.

A directory maps names to file identifiers/inodes.

A filesystem provides:

- logical abstractions: files, directories, paths, file descriptors.
- physical management: mapping file bytes to disk blocks/sectors.

Common system calls:

- `open()`: get a file descriptor.
- `read()`: copy file bytes into memory.
- `write()`: copy memory bytes to file.
- `fsync()`: force dirty data to disk.
- `close()`: release file descriptor state.
- `lseek()`: change file offset for random access.

### Databases vs Files

Every database ultimately stores bytes in files, but a database adds:

- a formal data model,
- schemas/metadata,
- indexing and query processing,
- integrity and reliability features,
- higher-level logical operations.

Data models:

- Relation: unordered rows and columns under a schema.
- Matrix: ordered numeric rows/columns.
- DataFrame: ordered rows/columns with labels and flexible column types.
- Tensor: multidimensional numeric array.
- Sequence/time series: ordered observations.
- Tree/graph: semistructured data, often serialized with JSON/XML-like formats.

### Row Store vs Column Store

Row-oriented layout:

- Stores all fields of a row together.
- Good when queries need most columns for selected rows.
- Wasteful when query needs only a few columns from many rows.

Column-oriented layout:

- Stores each column separately or in column chunks.
- Good for analytics that scan a few columns across many rows.
- Enables column pruning.
- Often compresses well.

Parquet:

- Common columnar data-lake file format.
- Stores metadata/statistics.
- Supports compression and column pruning.
- Often much less I/O than CSV/JSON for analytics.

### Useful Throughput

If a row-store file has many equal-size columns but query needs one column:

```text
useful data = total file size / number of columns
useful throughput = useful data / runtime
useful bandwidth percentage = useful throughput / device bandwidth
```

Example:

12 GB row-oriented file, 120 columns, query needs one column, runtime 30 s.

```text
useful data = 12 GB / 120 = 0.1 GB = 100 MB
useful throughput = 100 MB / 30 s = 3.33 MB/s
if SSD bandwidth = 600 MB/s:
useful percentage = 3.33 / 600 = 0.00556 = 0.556%
```

Interpretation: most I/O is wasted because the storage layout does not match the access pattern.

### Dataset Size and I/O Cost

Recipe:

1. Compute row size.
2. Multiply by row count.
3. Convert to GB as specified.
4. Multiply by number of passes, epochs, learning rates, reads, and writes.

Example:

500 million rows:

```text
UserID int64 = 8 bytes
Timestamp int64 = 8 bytes
Value float64 = 8 bytes
row size = 24 bytes
file size = 500e6 * 24 = 12e9 bytes = 12 GB
10 epochs * 3 learning rates = 30 full reads
total read I/O = 12 GB * 30 = 360 GB
```

Feature expansion example:

Dataset has `n = 200 million`, `d = 4` features plus label, all float64.

Pairwise interactions with no squared terms:

```text
new feature count = C(d, 2) = d(d - 1)/2 = 4*3/2 = 6
original columns = 4 features + 1 label = 5
expanded columns = 5 + 6 = 11
original size = 200e6 * 5 * 8 = 8 GB
expanded size = 200e6 * 11 * 8 = 17.6 GB
```

If model selection tries 3 learning rates and 4 epochs:

```text
passes = 3 * 4 = 12
Step A reads original = 8 * 12 = 96 GB
Step B reads original + writes expanded = 8 + 17.6 = 25.6 GB
Step C reads expanded = 17.6 * 12 = 211.2 GB
total I/O = 96 + 25.6 + 211.2 = 332.8 GB
```

### Disk I/O

For magnetic disks:

```text
T_IO = T_seek + T_rotation + T_transfer
```

Average rotational latency:

```text
rotation period = 60,000 ms / RPM
average rotational latency = rotation period / 2
```

Transfer time:

```text
T_transfer = data size / transfer bandwidth
```

Contiguous vs fragmented:

- Contiguous file: usually one/few seeks and rotational waits plus large sequential transfer.
- Fragmented file: many chunks, each may pay seek + rotation again.

Example:

2 GB CSV on 7200 RPM HDD, average seek 8 ms, transfer 150 MB/s, fragmented into 512 chunks.

```text
rotation period = 60000 / 7200 = 8.33 ms
average rotation = 4.17 ms
transfer = 2048 MB / 150 MB/s = 13.65 s
contiguous time ~= 8 ms + 4.17 ms + 13.65 s = 13.66 s
fragmented time ~= 512*(8 + 4.17) ms + 13.65 s
                ~= 6.23 s + 13.65 s = 19.88 s
slowdown ~= 19.88 / 13.66 = 1.46x
```

Interpretation: storage layout matters. For small random I/Os, seek and rotation dominate; for large contiguous scans, transfer bandwidth dominates.

## 9. Parallelism and Scalable Data Processing

### Why Large-Scale Data?

Large data can:

- lower variance in ML,
- enable personalization,
- reveal rare/granular patterns,
- support science, business, medicine, and search applications.

But large data requires scalable systems because one CPU/core/machine is often too slow or too small.

### Concurrency vs Parallelism

Concurrency: multiple tasks are in progress over the same time period. They may interleave on one core.

Parallelism: multiple tasks literally execute at the same time on different cores/processors/machines.

### Dataflow and Task Graphs

A dataflow graph represents operations and dependencies.

A task graph represents executable tasks and dependency edges.

Scheduling rule:

- A task can run only when its dependencies are done.
- Independent ready tasks can run in parallel.

Degree of parallelism:

```text
largest number of tasks that can run simultaneously
```

Critical path:

```text
longest dependency chain through the graph
```

Lower bound on completion time with `p` workers:

```text
T_p >= max(total_work / p, critical_path_length)
```

If a schedule completes in `T` time with `p` workers:

```text
total worker capacity = p * T
idle time = p*T - total_work
```

Example:

Total work = 100, critical path = 40, workers = 4.

```text
work/p = 100/4 = 25
lower bound = max(25, 40) = 40
maximum possible speedup = T1 / lower_bound = 100 / 40 = 2.5x
if actual completion = 50:
idle time = 4*50 - 100 = 100 worker-time units
```

### Speedup, Scaleup, Efficiency

Speedup:

```text
speedup = T_1 / T_p
```

Efficiency:

```text
efficiency = speedup / p
```

Example:

Runtime is 24 min on 1 worker and 5 min on 6 workers.

```text
speedup = 24 / 5 = 4.8x
efficiency = 4.8 / 6 = 0.8 = 80%
```

CPU utilization example:

Job spends 18 s computing and 27 s waiting on communication.

```text
utilization = 18 / (18 + 27) = 40%
```

### Amdahl's Law

If fraction `p` of work is parallelizable and `1-p` is serial:

```text
speedup(n) = 1 / ((1 - p) + p/n)
```

Infinite-core limit:

```text
speedup(infinity) = 1 / (1 - p)
```

Example:

25% serial, 75% parallel, 8 cores:

```text
infinite-core speedup = 1 / 0.25 = 4x
8-core speedup = 1 / (0.25 + 0.75/8)
               = 1 / 0.34375
               = 2.91x
```

Interpretation: speedup is sublinear because serial work remains.

Amdahl assumes fixed problem size. In real scalable data settings, larger datasets can make constant serial overhead relatively smaller, so measured scaling can differ.

### Dask

Dask is a Python library for parallel computing.

Important ideas:

- Lazy evaluation: operations build a graph instead of executing immediately.
- `compute()` triggers execution.
- Scheduler executes tasks across threads/processes/machines.
- Chunk/partition size matters.

Dask best-practice themes:

- Avoid too few partitions: low parallelism.
- Avoid too many tiny tasks: scheduling overhead dominates.
- Use diagnostics/dashboard to find bottlenecks.
- Prefer batching when task overhead is too high.
- Prefer breaking up work when tasks are too large or block parallelism.

### SIMD, GPUs, and NUMA

SIMD: Single Instruction, Multiple Data.

Example:

- AVX-512 has 512-bit registers.
- It can process 16 float32 values or 8 float64 values per instruction.

SIMT/SPMD generalize the idea to many threads executing similar programs over different data.

GPU ideas:

- Many lightweight threads.
- Good for high arithmetic intensity.
- Bad when data transfer, branching, or memory access dominates.

GPU bottlenecks:

- Host-device transfer overhead, such as PCIe.
- Limited VRAM.
- Thread divergence: threads in a warp take different branches.
- Low arithmetic intensity: too few FLOPs per byte loaded.
- Communication overhead across GPUs, such as gradient synchronization.

NUMA:

- In multi-socket machines, local memory is faster than remote memory.
- If remote access is 3x and local latency is 100 ns, remote latency is 300 ns.

False sharing:

- Two threads write different variables that sit on the same cache line.
- Cores keep invalidating each other's cache line even though the variables are logically separate.
- Fix by padding, aligning, or partitioning data so threads do not write the same cache line.

## 10. High-Yield Problem Templates

### Template A: Bits Needed

Question: minimum bits for `N` unique items.

```text
Find smallest k such that 2^k >= N.
```

Example:

```text
N = 70,000
2^16 = 65,536
2^17 = 131,072
answer = 17 bits
```

### Template B: Instruction Alignment

Question: valid instruction addresses and fraction of valid byte addresses.

```text
instruction size = alignment = b bytes
next addresses advance by b
valid fraction = 1/b
```

For 8-byte aligned instructions:

```text
low 3 bits fixed at 000
valid fraction = 1/8
```

### Template C: CPI and Runtime

Question: instruction mix and clock speed.

```text
CPI = weighted average
cycles = instructions * CPI
time = cycles / clock
```

With stalls:

```text
CPI_with_stalls = base_CPI + memory_fraction * stall_cycles
```

or compute total base cycles and stall cycles separately.

### Template D: Memory Stalls as Percentage

Question: what percent of total cycles are stalls?

```text
stall_percent = stall_cycles / total_cycles
```

Example:

1B instructions, 25% memory ops, 40-cycle stall, base 1 cycle each.

```text
base cycles = 1B
memory ops = 0.25B
stall cycles = 0.25B * 40 = 10B
total cycles = 11B
stall percent = 10/11 = 90.9%
```

If stall reduces to 10:

```text
new stall = 0.25B * 10 = 2.5B
new total = 1B + 2.5B = 3.5B
speedup = 11B / 3.5B = 3.14x
```

### Template E: AMAT

Question: cache hierarchy average access time.

```text
miss_rate = 1 - hit_rate
AMAT = L1 + L1_miss * (L2 + L2_miss * DRAM)
```

Remember: if L2 hit rate is "given L1 miss," multiply DRAM probability by both misses.

### Template F: Cache Line Miss Fraction

Question: sequential array scan.

```text
elements_per_line = cache_line_bytes / element_bytes
miss_fraction ~= 1 / elements_per_line
hit_fraction ~= (elements_per_line - 1) / elements_per_line
```

For 64B line and 8B float64:

```text
8 elements per line
miss fraction ~= 1/8
```

### Template G: Page Counts

Question: how many pages for an array/file.

```text
pages = ceil(data_size / page_size)
```

Example:

```text
1 MB array, 4 KB pages
1 MB = 1024 KB
pages = 1024 / 4 = 256
```

### Template H: Page Table Size

```text
offset bits = log2(page_size)
VPN bits = address bits - offset bits
entries = 2^(VPN bits)
page table bytes = entries * PTE bytes
```

### Template I: Scheduling

Use a table:

```text
process | arrival | burst | first run | completion | turnaround | response
```

Then:

```text
turnaround = completion - arrival
response = first run - arrival
average = sum / count
```

For Round Robin:

- Add jobs as they arrive.
- Run current job for min(quantum, remaining time).
- If unfinished, append to back.
- Completion time is when remaining time reaches zero.
- Response time uses first time the job ever ran.

### Template J: Row-Store Wasted I/O

```text
useful data = total file size * needed_columns / total_columns
useful throughput = useful data / time
useful bandwidth % = useful throughput / device bandwidth
```

### Template K: Feature Expansion I/O

For pairwise interactions with no squared terms:

```text
new columns = C(d, 2) = d(d - 1)/2
expanded columns = original feature columns + label columns + new columns
file size = rows * columns * bytes_per_value
total I/O = sum(all reads and writes)
```

### Template L: Task Graph Bounds

```text
best possible T_p >= max(total_work / p, critical_path)
speedup <= total_work / lower_bound
idle_time = p*actual_time - total_work
```

### Template M: Amdahl's Law

```text
speedup(n) = 1 / ((1-p) + p/n)
max speedup = 1 / (1-p)
```

### Template N: Disk I/O

```text
rotation period ms = 60000 / RPM
average rotation ms = period / 2
transfer time = size / bandwidth
T_IO = seek + rotation + transfer
fragmented T_IO = chunks*(seek + rotation) + transfer
```

## 11. Common Mistakes to Avoid

- Mixing bits and bytes.
- Forgetting that a byte-addressable address points to one byte.
- Using decimal GB when the problem expects `2^30`, or using `2^30` when the problem explicitly says `10^9`.
- Forgetting that signed 8-bit max is 127, not 255.
- Saying overflow just because the top bit is 1; the real signed-addition rule is same-sign inputs producing opposite-sign output.
- Treating L2 hit rate as global when it is given conditional on L1 miss.
- Forgetting to update recency on LRU hits.
- Forgetting context switch overhead in scheduling.
- Computing response time as completion time; response is first run minus arrival.
- Counting row-store file size as useful data when only one column is needed.
- Ignoring writes in I/O cost questions.
- Assuming speedup is linear with workers.
- Forgetting critical path in task-graph lower bounds.
- Optimizing GPU/parallel code before checking whether memory transfer or serial work dominates.

## 12. Final Exam-Style Checklist

You are ready when you can do these without notes:

- Convert among binary, hex, octal, and decimal.
- Compute bits needed for unique items.
- Identify valid aligned addresses and address fractions.
- Detect signed two's-complement overflow and interpret wrapped bit patterns.
- Evaluate a small custom floating-point bit string.
- Compute CPI, cycles, runtime, stall percentage, and speedup.
- Compute two-level and three-level AMAT.
- Explain temporal vs spatial locality.
- Predict row-major vs column-wise cache behavior.
- Explain why NumPy/vectorized code can beat Python loops.
- Define OS abstraction, mechanism, and policy.
- Simulate FCFS, SJF, SRTF, and Round Robin.
- Compute turnaround, waiting, response, and context switch overhead.
- Translate virtual to physical addresses using VPN/offset/PFN.
- Compute page table entries and page table memory.
- Simulate FIFO/LRU page replacement.
- Explain thrashing and page-size tradeoffs.
- Distinguish files, directories, filesystems, databases, and data models.
- Compute dataset size and full-scan I/O across epochs/learning rates.
- Compare row-store vs column-store and explain Parquet's advantages.
- Compute disk seek/rotation/transfer time and fragmentation cost.
- Compute task graph lower bounds, speedup, efficiency, and idle time.
- Apply Amdahl's law.
- Explain Dask lazy task graphs and partition-size tradeoffs.
- Explain SIMD, GPU bottlenecks, NUMA, and false sharing.

## 13. One-Page Formula Sheet

```text
bits for N items = ceil(log2 N)
k bits -> 2^k values
byte = 8 bits

unsigned k-bit range = 0 to 2^k - 1
signed two's-complement range = -2^(k-1) to 2^(k-1)-1
signed value when MSB=1 = unsigned_value - 2^k

CPI = sum(f_i * cycles_i)
cycles = instruction_count * CPI
time = cycles / clock_rate
speedup = old_time / new_time

stall_cycles = memory_ops * stall_per_memory_op
stall_percent = stall_cycles / total_cycles

AMAT two-level = hit_time + miss_rate * miss_penalty
AMAT three-level = L1 + L1_miss * (L2 + L2_miss * DRAM)
P(DRAM) = L1_miss * L2_miss

elements_per_cache_line = cache_line_bytes / element_bytes
sequential miss fraction ~= 1 / elements_per_cache_line

offset bits = log2(page_size)
VPN bits = virtual_address_bits - offset_bits
page_table_entries = 2^(VPN bits)
page_table_size = entries * PTE_size
pages = ceil(data_size / page_size)

turnaround = completion - arrival
response = first_run - arrival
waiting = turnaround - burst   # if no extra blocked time; handle context switches carefully

row_size = sum(column_sizes)
file_size = rows * row_size
passes = epochs * hyperparameter_trials
read_IO = file_size * passes

useful_data = file_size * needed_columns / total_columns
useful_throughput = useful_data / runtime

rotation_period_ms = 60000 / RPM
avg_rotation_ms = rotation_period_ms / 2
disk_IO = seek + rotation + transfer
transfer = data_size / bandwidth

parallel lower bound = max(total_work / workers, critical_path)
parallel speedup = T1 / Tp
efficiency = speedup / workers
idle_time = workers * actual_time - total_work

Amdahl speedup(n) = 1 / ((1-p) + p/n)
Amdahl max speedup = 1 / (1-p)
```
