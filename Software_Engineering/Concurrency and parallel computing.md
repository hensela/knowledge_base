In Python, different names (thread, task, process) are used to describe *things are are occurring simultaneously*... but what does *simultaneous* technically mean? We provide a high-level overview of concurrency and parallel computing and a quick review of `threading`, `asyncio`, `multiprocessing` and their main differences. 

## Must know
Let's assume we need to perform some tasks: T1, T2, T3, T4.  
The simplest idea is to complete them **sequentially**, making sure we complete one step before moving on to the next one: T1 -> T2 -> T3 -> T4.  

However, there may be some waiting times within each task (e.g., ask for external inputs before proceding) and the dependencies may not be that strict, so we could do something like: 50% T2 -> T1 -> 30% T3 -> 50% T2 -> T4 -> 70% T3.  
We're still operating sequentially, but we're optimizing the switches between tasks so that the total time will be lower. This is what  `threading` does by creating **threads** to handle tasks and deciding when to switch from one to another.  
If we have an interrupt mechanism that can suspend the currently executing task (any task, at any time!) and a scheduler to determine the new task to switch to, this is called **preemptive multitasking** and that's exactly what's done by  `threading`.  
If the **tasks** are in full control over their execution (meaning they can't be stopped unless they explicitly tell so), this is called **cooperative multitasking** and it's offered by `asyncio`. Such a framework allows for a more granular control over task switching and interdependencies, but the coding complexity may increase greatly.  
As a rule of thumb, if we don't face [risks](#corner-cases) in switching tasks we're probably better off with preemptive multitasking.  

So far we've only been talking about running and switching tasks: even though our overall process is arguably optimized and we can start multiple tasks in parallel, we're still tackling them one at a time. In other words, our execution is still sequential and we only need one processor (CPU).  
Proper **parallel computing** is a type of concurrency that leverages multiple CPUs so that the tasks can really be operated at the same time, as in CPU1: T1 -> T2, CPU2: T3 -> T4. Of course, this is only possible if T3 doesn't depend on T1 and T2.  
Whenever we're dealing with multiple independent tasks with long computation times, `multiprocessing` is the best way to go. Programs facing these challenges are called **CPU-bound** (e.g., we need to fit a heavy ML model), while the ones wher the tasks have long waiting times are called **I/O-bound** (e.g., we need to query Hive). 


|Package|Bound by|Concurrency|CPUs|Terminology|
|:---:|:---:|:---:|:---:|:---:|
|`threading`| I/O | preemptive multitasking | 1  | thread  |
| `asyncio`  | I/O | collaborative multitasking | 1  |  task |
|`multiprocessing`| CPU | multiprocessing  |  Many |  process |

## Details
#### Corner cases 
Some concurrency types may not be suited to speed up specific processes. For instance, we may struggle with ([race conditions](https://en.wikipedia.org/wiki/Race_condition#Software)) whenever different threads need to access a common resource. Moreover, creating threads and processes comes with overheads that may lead us to slower running times even compared to the naive sequential processing.  
When implementing concurrency we must think about the main limiting factors we want to deal with and the associated risks, especially because we may introduce bugs that are very difficult to spot and reproduce. 

#### Additional resources/info
* [Concurrency (Real Python)](https://realpython.com/python-concurrency/)  
* [Async IO Walkthrough (Real Python)](https://realpython.com/async-io-python/)
* [Multithreading & multiprocessing](https://stackoverflow.com/questions/27455155/python-multiprocessing-combined-with-multithreading)
