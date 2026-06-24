# Modal Rules and Guidelines for LLMs

This file provides rules and guidelines for LLMs when implementing Modal code.

## General

- Modal is a serverless cloud platform for running Python code with minimal configuration
- Designed for AI/ML workloads but supports general-purpose cloud compute
- Serverless billing model - you only pay for resources used

## Modal documentation

- Extensive documentation is available at: modal.com/docs (and in markdown format at modal.com/llms-full.txt)
- A large collection of examples is available at: modal.com/docs/examples (and github.com/modal-labs/modal-examples)
- Reference documentation is available at: modal.com/docs/reference

Always refer to documentation and examples for up-to-date functionality and exact syntax.

## Core Modal concepts

### App

- A group of functions, classes and sandboxes that are deployed together.

### Function

- The basic unit of serverless execution on Modal.
- Each Function executes in its own container, and you can configure different Images for different Functions within the same App:

  ```python
  image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers")
    .apt_install("ffmpeg")
    .run_commands("mkdir -p /models")
  )

  @app.function(image=image)
  def square(x: int) -> int:
    return x * x
  ```

- You can configure individual hardware requirements (CPU, memory, GPUs, etc.) for each Function.

  ```python
  @app.function(
    gpu="H100",
    memory=4096,
    cpu=2,
  )
  def inference():
    ...
  ```

  Some examples specificly for GPUs:

  ```python
  @app.function(gpu="A10G")  # Single GPU, e.g. T4, A10G, A100, H100, or "any"
  @app.function(gpu="A100:2")  # Multiple GPUs, e.g. 2x A100 GPUs
  @app.function(gpu=["H100", "A100", "any"]) # GPU with fallbacks
  ```

- Functions can be invoked in a number of ways. Some of the most common are:
  - `foo.remote()` - Run the Function in a separate container in the cloud. This is by far the most common.
  - `foo.local()` - Run the Function in the same context as the caller. Note: This does not necessarily mean locally on your machine.
  - `foo.map()` - Parallel map over a set of inputs.
  - `foo.spawn()` - Calls the function with the given arguments, without waiting for the results. Terminating the App will also terminate spawned functions.
- Web endpoint: You can turn any Function into an HTTP web endpoint served by adding a decorator:

  ```python
  @app.function()
  @modal.fastapi_endpoint()
  def fastapi_endpoint():
    return {"status": "ok"}

  @app.function()
  @modal.asgi_app()
  def asgi_app():
    app = FastAPI()
    ...
    return app
  ```

- You can run Functions on a schedule using e.g. `@app.function(schedule=modal.Period(minutes=5))` or `@app.function(schedule=modal.Cron("0 9 * * *"))`.

### Classes (a.k.a. `Cls`)

- For stateful operations with startup/shutdown lifecycle hooks. Example:

  ```python
  @app.cls(gpu="A100")
  class ModelServer:
      @modal.enter()
      def load_model(self):
          # Runs once when container starts
          self.model = load_model()

      @modal.method()
      def predict(self, text: str) -> str:
          return self.model.generate(text)

      @modal.exit()
      def cleanup(self):
          # Runs when container stops
          cleanup()
  ```

### Other important concepts

- Image: Represents a container image that Functions can run in.
- Sandbox: Allows defining containers at runtime and securely running arbitrary code inside them.
- Volume: Provide a high-performance distributed file system for your Modal applications.
- Secret: Enables securely providing credentials and other sensitive information to your Modal Functions.
- Dict: Distributed key/value store, managed by Modal.
- Queue: Distributed, FIFO queue, managed by Modal.

## Differences from standard Python development

- Modal always executes code in the cloud, even while you are developing. You can use Environments for separating development and production deployments.
- Dependencies: It's common and encouraged to have different dependency requirements for different Functions within the same App. Consider defining dependencies in Image definitions (see Image docs) that are attached to Functions, rather than in global `requirements.txt`/`pyproject.toml` files, and putting `import` statements inside the Function `def`. Any code in the global scope needs to be executable in all environments where that App source will be used (locally, and any of the Images the App uses).

## Modal coding style

- Modal Apps, Volumes, and Secrets should be named using kebab-case.
- Always use `import modal`, and qualified names like `modal.App()`, `modal.Image.debian_slim()`.
- Modal evolves quickly, and prints helpful deprecation warnings when you `modal run` an App that uses deprecated features. When writing new code, never use deprecated features.

## Common commands

Running `modal --help` gives you a list of all available commands. All commands also support `--help` for more details.

### Running your Modal app during development

- `modal run path/to/your/app.py` - Run your app on Modal.
- `modal run -m module.path.to.app` - Run your app on Modal, using the Python module path.
- `modal serve modal_server.py` - Run web endpoint(s) associated with a Modal app, and hot-reload code on changes. Will print a URL to the web endpoint(s). Note: you need to use `Ctrl+C` to interrupt `modal serve`.

### Deploying your Modal app

- `modal deploy path/to/your/app.py` - Deploy your app (Functions, web endpoints, etc.) to Modal.
- `modal deploy -m module.path.to.app` - Deploy your app to Modal, using the Python module path.

Logs:

- `modal app logs <app_name>` - Stream logs for a deployed app. Note: you need to use `Ctrl+C` to interrupt the stream.

### Resource management

- There are CLI commands for interacting with resources like `modal app list`, `modal volume list`, and similarly for `secret`, `dict`, `queue`, etc.
- These also support other command than `list` - use e.g. `modal app --help` for more.

## Testing and debugging

- When using `app.deploy()`, you can wrap it in a `with modal.enable_output():` block to get more output.

# Scaling out

Modal makes it trivially easy to scale compute across thousands of containers.
You won't have to worry about your App crashing if it goes viral or need to wait
a long time for your batch jobs to complete.

For the the most part, scaling out will happen automatically, and you won't need
to think about it. But it can be helpful to understand how Modal's autoscaler
works and how you can control its behavior when you need finer control.

## How does autoscaling work on Modal?

Every Modal Function corresponds to an autoscaling pool of containers. The size
of the pool is managed by Modal's autoscaler. The autoscaler will spin up new
containers when there is no capacity available for new inputs, and it will spin
down containers when resources are idling. By default, Modal Functions will
scale to zero when there are no inputs to process.

Autoscaling decisions are made quickly and frequently so that your batch jobs
can ramp up fast and your deployed Apps can respond to any sudden changes in
traffic.

## Configuring autoscaling behavior

Modal exposes a few settings that allow you to configure the autoscaler's
behavior. These settings can be passed to the `@app.function` or `@app.cls`
decorators:

- `max_containers`: The upper limit on containers for the specific Function.
- `min_containers`: The minimum number of containers that should be kept warm,
  even when the Function is inactive.
- `buffer_containers`: The size of the buffer to maintain while the Function is
  active, so that additional inputs will not need to queue for a new container.
- `scaledown_window`: The maximum duration (in seconds) that individual
  containers can remain idle when scaling down.

In general, these settings allow you to trade off cost and latency. Maintaining
a larger warm pool or idle buffer will increase costs but reduce the chance that
inputs will need to wait for a new container to start.

Similarly, a longer scaledown window will let containers idle for longer, which
might help avoid unnecessary churn for Apps that receive regular but infrequent
inputs. Note that containers may not wait for the entire scaledown window before
shutting down if the App is substantially overprovisioned.

## Dynamic autoscaler updates

It's also possible to update the autoscaler settings dynamically (i.e., without redeploying
the App) using the [`Function.update_autoscaler()`](/docs/reference/modal.Function#update_autoscaler)
method:

```python notest
f = modal.Function.from_name("my-app", "f")
f.update_autoscaler(max_containers=100)
```

The autoscaler settings will revert to the configuration in the function
decorator the next time you deploy the App. Or they can be overridden by
further dynamic updates:

```python notest
f.update_autoscaler(min_containers=2, max_containers=10)
f.update_autoscaler(min_containers=4)  # max_containers=10 will still be in effect
```

A common pattern is to run this method in a [scheduled function](/docs/guide/cron)
that adjusts the size of the warm pool (or container buffer) based on the time of day:

```python
@app.function()
def inference_server():
    ...

@app.function(schedule=modal.Cron("0 6 * * *", timezone="America/New_York"))
def increase_warm_pool():
    inference_server.update_autoscaler(min_containers=4)

@app.function(schedule=modal.Cron("0 22 * * *", timezone="America/New_York"))
def decrease_warm_pool():
    inference_server.update_autoscaler(min_containers=0)
```

When you have a [`modal.Cls`](/docs/reference/modal.Cls), `update_autoscaler`
is a method on an _instance_ and will control the autoscaling behavior of
containers serving the Function with that specific set of parameters:

```python notest
MyClass = modal.Cls.from_name("my-app", "MyClass")
obj = MyClass(model_version="3.5")
obj.update_autoscaler(buffer_containers=2)  # type: ignore
```

Note that it's necessary to disable type checking on this line, because the
object will appear as an instance of the class that you defined rather than the
Modal wrapper type.

## Parallel execution of inputs

If your code is running the same function repeatedly with different independent
inputs (e.g., a grid search), the easiest way to increase performance is to run
those function calls in parallel using Modal's
[`Function.map()`](/docs/reference/modal.Function#map) method.

Here is an example if we had a function `evaluate_model` that takes a single
argument:

```python
import modal

app = modal.App()


@app.function()
def evaluate_model(x):
    ...


@app.local_entrypoint()
def main():
    inputs = list(range(100))
    for result in evaluate_model.map(inputs):  # runs many inputs in parallel
        ...
```

In this example, `evaluate_model` will be called with each of the 100 inputs
(the numbers 0 - 99 in this case) roughly in parallel and the results are
returned as an iterable with the results ordered in the same way as the inputs.

### Exceptions

By default, if any of the function calls raises an exception, the exception will
be propagated. To treat exceptions as successful results and aggregate them in
the results list, pass in
[`return_exceptions=True`](/docs/reference/modal.Function#map).

```python
@app.function()
def my_func(a):
    if a == 2:
        raise Exception("ohno")
    return a ** 2

@app.local_entrypoint()
def main():
    print(list(my_func.map(range(3), return_exceptions=True, wrap_returned_exceptions=False)))
    # [0, 1, Exception('ohno'))]
```

Note: prior to version 1.0.5, the returned exceptions inadvertently leaked an internal
wrapper type (`modal.exceptions.UserCodeException`). To avoid breaking any user code that
was checking exception types, we're taking a gradual approach to fixing this bug. Adding
`wrap_returned_exceptions=False` will opt-in to the future default behavior and return the
underlying exception type without a wrapper.

### Starmap

If your function takes multiple variable arguments, you can either use
[`Function.map()`](/docs/reference/modal.Function#map) with one input iterator
per argument, or [`Function.starmap()`](/docs/reference/modal.Function#starmap)
with a single input iterator containing sequences (like tuples) that can be
spread over the arguments. This works similarly to Python's built in `map` and
`itertools.starmap`.

```python
@app.function()
def my_func(a, b):
    return a + b

@app.local_entrypoint()
def main():
    assert list(my_func.starmap([(1, 2), (3, 4)])) == [3, 7]
```

### Gotchas

Note that `.map()` is a method on the modal function object itself, so you don't
explicitly _call_ the function.

Incorrect usage:

```python notest
results = evaluate_model(inputs).map()
```

Modal's map is also not the same as using Python's builtin `map()`. While the
following will technically work, it will execute all inputs in sequence rather
than in parallel.

Incorrect usage:

```python notest
results = map(evaluate_model, inputs)
```

## Asynchronous usage

All Modal APIs are available in both blocking and asynchronous variants. If you
are comfortable with asynchronous programming, you can use it to create
arbitrary parallel execution patterns, with the added benefit that any Modal
functions will be executed remotely. See the [async guide](/docs/guide/async) or
the examples for more information about asynchronous usage.

## GPU acceleration

Sometimes you can speed up your applications by utilizing GPU acceleration. See
the [GPU section](/docs/guide/gpu) for more information.

## Scaling Limits

Modal enforces the following limits for every function:

- 2,000 pending inputs (inputs that haven't been assigned to a container yet)
- 25,000 total inputs (which include both running and pending inputs)

For inputs created with `.spawn()` for async jobs, Modal allows up to 1 million pending inputs instead of 2,000.

If you try to create more inputs and exceed these limits, you'll receive a `Resource Exhausted` error, and you should retry your request later. If you need higher limits, please reach out!

Additionally, each `.map()` invocation can process at most 1000 inputs concurrently.

# Input concurrency



This guide documents the use of the `modal.concurrent` decorator to
process multiple inputs at the same time in a single Modal container.

This page is a high-level guide to input concurrency. For reference documentation
of the `modal.concurrent` decorator, see [this page](/docs/reference/modal.concurrent).

## Overview

As traffic to your application increases, Modal will automatically scale up the
number of containers running your Function:

<div class="flex justify-center"><NoConcurrentInputs /></div>

By default, each container will be assigned one input at a time. Autoscaling
across containers allows your Function to process inputs in parallel. This is
ideal when the operations performed by your Function are CPU-bound.

For some workloads, though, it is inefficient for containers to process inputs
one-by-one. Modal supports these workloads with its _input concurrency_ feature,
which allows individual containers to process multiple inputs at the same time:

<div class="flex justify-center"><WithConcurrentInputs /></div>

When used effectively, input concurrency can reduce latency and lower costs.

## Use cases

Input concurrency can be especially effective for workloads that are primarily
I/O-bound, e.g.:

- Querying a database
- Making external API requests
- Making remote calls to other Modal Functions

For such workloads, individual containers may be able to concurrently process
large numbers of inputs with minimal additional latency. This means that your
Modal application will be more efficient overall, as it won't need to scale
containers up and down as traffic ebbs and flows.

Another use case is to leverage _continuous batching_ on GPU-accelerated
containers. Frameworks such as [vLLM](/docs/examples/llm_inference) can
achieve the benefits of batching across multiple inputs even when those
inputs do not arrive simultaneously (because new batches are formed for each
forward pass of the model).

Note that for CPU-bound workloads, input concurrency will likely not be as
effective (or will even be counterproductive), and you may want to use
Modal's [_dynamic batching_ feature](/docs/guide/dynamic-batching) instead.

## Enabling input concurrency

To enable input concurrency, add the `@modal.concurrent` decorator:

```python
@app.function()
@modal.concurrent(max_inputs=100)
def my_function(input: str):
    ...

```

When using the class pattern, the decorator should be applied at the level of
the _class_, not on individual methods:

```python
@app.cls()
@modal.concurrent(max_inputs=100)
class MyCls:

    @modal.method()
    def my_method(self, input: str):
        ...
```

Because all methods on a class will be served by the same containers, a class
with input concurrency enabled will concurrently run distinct methods in
addition to multiple inputs for the same method.

## Setting a concurrency target

When using the `@modal.concurrent` decorator, you must always configure the
maximum number of inputs that each container will concurrently process. If
demand exceeds this limit, Modal will automatically scale up more containers.

Additional inputs may need to queue up while these additional containers cold
start. To help avoid degraded latency during scaleup, the `@modal.concurrent`
decorator has a separate `target_inputs` parameter. When set, Modal's autoscaler
will aim for this target as it provisions resources. If demand increases faster
than new containers can spin up, the active containers will be allowed to burst
above the target up to the `max_inputs` limit:

```python
@app.function()
@modal.concurrent(max_inputs=96, target_inputs=80)  # Allow a 20% burst
def my_function(input: str):
    ...
```

It may take some experimentation to find the right settings for these parameters
in your particular application. Our suggestion is to set the `target_inputs`
based on your desired latency and the `max_inputs` based on resource constraints
(i.e., to avoid GPU OOM). You may also consider the relative latency cost of
scaling up a new container versus overloading the existing containers.

## Concurrency mechanisms

Modal uses different concurrency mechanisms to execute your Function depending
on whether it is defined as synchronous or asynchronous. Each mechanism imposes
certain requirements on the Function implementation. Input concurrency is an
advanced feature, and it's important to make sure that your implementation
complies with these requirements to avoid unexpected behavior.

For synchronous Functions, Modal will execute concurrent inputs on separate
threads. _This means that the Function implementation must be thread-safe._

```python
# Each container can execute up to 10 inputs in separate threads
@app.function()
@modal.concurrent(max_inputs=10)
def sleep_sync():
    # Function must be thread-safe
    time.sleep(1)
```

For asynchronous Functions, Modal will execute concurrent inputs using
separate `asyncio` tasks on a single thread. This does not require thread
safety, but it does mean that the Function needs to participate in
collaborative multitasking (i.e., it should not block the event loop).

```python
# Each container can execute up to 10 inputs with separate async tasks
@app.function()
@modal.concurrent(max_inputs=10)
async def sleep_async():
    # Function must not block the event loop
    await asyncio.sleep(1)
```

## Gotchas

Input concurrency is a powerful feature, but there are a few caveats that can
be useful to be aware of before adopting it.

### Input cancellations

Synchronous and asynchronous Functions handle input cancellations differently.
Modal will raise a `modal.exception.InputCancellation` exception in synchronous
Functions and an `asyncio.CancelledError` in asynchronous Functions.

When using input concurrency with a synchronous Function, a single input
cancellation will terminate the entire container. If your workflow depends on
graceful input cancellations, we recommend using an asynchronous
implementation.

### Concurrent logging

The separate threads or tasks that are executing the concurrent inputs will
write any logs to the same stream. This makes it difficult to associate logs
with a specific input, and filtering for a specific function call in Modal's web
dashboard will show logs for all inputs running at the same time.

To work around this, we recommend including a unique identifier in the messages
you log (either your own identifier or the `modal.current_input_id()`) so that
you can use the search functionality to surface logs for a specific input:

```python
@app.function()
@modal.concurrent(max_inputs=10)
async def better_concurrent_logging(x: int):
    logger.info(f"{modal.current_input_id()}: Starting work with {x}")
```

# Batch Processing

Modal is optimized for large-scale batch processing, allowing functions to scale to thousands of parallel containers with zero additional configuration. Function calls can be submitted asynchronously for background execution, eliminating the need to wait for jobs to finish or tune resource allocation.

This guide covers Modal's batch processing capabilities, from basic invocation to integration with existing pipelines.

## Background Execution with `.spawn_map`

The fastest way to submit multiple jobs for asynchronous processing is by invoking a function with `.spawn_map`. When combined with the [`--detach`](/docs/reference/cli/run) flag, your App continues running until all jobs are completed.

Here's an example of submitting 100,000 videos for parallel embedding. You can disconnect after submission, and the processing will continue to completion in the background:

```python
# Kick off asynchronous jobs with `modal run --detach batch_processing.py`
import modal

app = modal.App("batch-processing-example")
volume = modal.Volume.from_name("video-embeddings", create_if_missing=True)

@app.function(volumes={"/data": volume})
def embed_video(video_id: int):
    # Business logic:
    # - Load the video from the volume
    # - Embed the video
    # - Save the embedding to the volume
    ...

@app.local_entrypoint()
def main():
    embed_video.spawn_map(range(100_000))
```

This pattern works best for jobs that store results externally—for example, in a [Modal Volume](/docs/guide/volumes), [Cloud Bucket Mount](/docs/guide/cloud-bucket-mounts), or your own database\*.

_\* For database connections, consider using [Modal Proxy](/docs/guide/proxy-ips) to maintain a static IP across thousands of containers._

## Parallel Processing with `.map`

Using `.map` allows you to offload expensive computations to powerful machines while gathering results. This is particularly useful for pipeline steps with bursty resource demands. Modal handles all infrastructure provisioning and de-provisioning automatically.

Here's how to implement parallel video similarity queries as a single Modal function call:

```python
# Run jobs and collect results with `modal run gather.py`
import modal

app = modal.App("gather-results-example")

@app.function(gpu="L40S")
def compute_video_similarity(query: str, video_id: int) -> tuple[int, int]:
    # Embed video with GPU acceleration & compute similarity with query
    return video_id, score


@app.local_entrypoint()
def main():
    import itertools

    queries = itertools.repeat("Modal for batch processing")
    video_ids = range(100_000)

    for video_id, score in compute_video_similarity.map(queries, video_ids):
        # Process results (e.g., extract top 5 most similar videos)
        pass
```

This example runs `compute_video_similarity` on an autoscaling pool of L40S GPUs, returning scores to a local process for further processing.

## Integration with Existing Systems

The recommended way to use Modal Functions within your existing data pipeline is through [deployed function invocation](/docs/guide/trigger-deployed-functions). After deployment, you can call Modal functions from external systems:

```python
def external_function(inputs):
    compute_similarity = modal.Function.from_name(
        "gather-results-example",
        "compute_video_similarity"
    )
    for result in compute_similarity.map(inputs):
        # Process results
        pass
```

You can invoke Modal Functions from any Python context, gaining access to built-in observability, resource management, and GPU acceleration.

# Job processing

Modal can be used as a scalable job queue to handle asynchronous tasks submitted
from a web app or any other Python application. This allows you to offload up to 1 million
long-running or resource-intensive tasks to Modal, while your main application
remains responsive.

## Creating jobs with .spawn()

The basic pattern for using Modal as a job queue involves three key steps:

1. Defining and deploying the job processing function using `modal deploy`.
2. Submitting a job using
   [`modal.Function.spawn()`](/docs/reference/modal.Function#spawn)
3. Polling for the job's result using
   [`modal.FunctionCall.get()`](/docs/reference/modal.FunctionCall#get)

Here's a simple example that you can run with `modal run my_job_queue.py`:

```python
# my_job_queue.py
import modal

app = modal.App("my-job-queue")

@app.function()
def process_job(data):
    # Perform the job processing here
    return {"result": data}

def submit_job(data):
    # Since the `process_job` function is deployed, need to first look it up
    process_job = modal.Function.from_name("my-job-queue", "process_job")
    call = process_job.spawn(data)
    return call.object_id

def get_job_result(call_id):
    function_call = modal.FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=5)
    except modal.exception.OutputExpiredError:
        result = {"result": "expired"}
    except TimeoutError:
        result = {"result": "pending"}
    return result

@app.local_entrypoint()
def main():
    data = "my-data"

    # Submit the job to Modal
    call_id = submit_job(data)
    print(get_job_result(call_id))
```

In this example:

- `process_job` is the Modal function that performs the actual job processing.
  To deploy the `process_job` function on Modal, run
  `modal deploy my_job_queue.py`.
- `submit_job` submits a new job by first looking up the deployed `process_job`
  function, then calling `.spawn()` with the job data. It returns the unique ID
  of the spawned function call.
- `get_job_result` attempts to retrieve the result of a previously submitted job
  using [`FunctionCall.from_id()`](/docs/reference/modal.FunctionCall#from_id) and
  [`FunctionCall.get()`](/docs/reference/modal.FunctionCall#get).
  [`FunctionCall.get()`](/docs/reference/modal.FunctionCall#get) waits indefinitely
  by default. It takes an optional timeout argument that specifies the maximum
  number of seconds to wait, which can be set to 0 to poll for an output
  immediately. Here, if the job hasn't completed yet, we return a pending
  response.
- The results of a `.spawn()` are accessible via `FunctionCall.get()` for up to
  7 days after completion. After this period, we return an expired response.

[Document OCR Web App](/docs/examples/doc_ocr_webapp) is an example that uses
this pattern.

## Integration with web frameworks

You can easily integrate the job queue pattern with web frameworks like FastAPI.
Here's an example, assuming that you have already deployed `process_job` on
Modal with `modal deploy` as above. This example won't work if you haven't
deployed your app yet.

```python
# my_job_queue_endpoint.py
import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App("fastapi-modal", image=image)


@app.function()
@modal.asgi_app()
@modal.concurrent(max_inputs=20)
def fastapi_app():
    from fastapi import FastAPI

    web_app = FastAPI()

    @web_app.post("/submit")
    async def submit_job_endpoint(data):
        process_job = modal.Function.from_name("my-job-queue", "process_job")

        call = await process_job.spawn.aio(data)
        return {"call_id": call.object_id}


    @web_app.get("/result/{call_id}")
    async def get_job_result_endpoint(call_id: str):
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            result = await function_call.get.aio(timeout=0)
        except modal.exception.OutputExpiredError:
            return fastapi.responses.JSONResponse(content="", status_code=404)
        except TimeoutError:
            return fastapi.responses.JSONResponse(content="", status_code=202)

        return result

    return web_app
```

In this example:

- The `/submit` endpoint accepts job data, submits a new job using
  `await process_job.spawn.aio()`, and returns the job's ID to the client.
- The `/result/{call_id}` endpoint allows the client to poll for the job's
  result using the job ID. If the job hasn't completed yet, it returns a 202
  status code to indicate that the job is still being processed. If the job
  has expired, it returns a 404 status code to indicate that the job is not found.

You can try this app by serving it with `modal serve`:

```shell
modal serve my_job_queue_endpoint.py
```

Then interact with its endpoints with `curl`:

```shell
# Make a POST request to your app endpoint with.
$ curl -X POST $YOUR_APP_ENDPOINT/submit?data=data
{"call_id":"fc-XXX"}

# Use the call_id value from above.
$ curl -X GET $YOUR_APP_ENDPOINT/result/fc-XXX
```

## Scaling and reliability

Modal automatically scales the job queue based on the workload, spinning up new
instances as needed to process jobs concurrently. It also provides built-in
reliability features like automatic retries and timeout handling.

You can customize the behavior of the job queue by configuring the
`@app.function()` decorator with options like
[`retries`](/docs/guide/retries#function-retries),
[`timeout`](/docs/guide/timeouts#timeouts), and
[`max_containers`](/docs/guide/scale#configuring-autoscaling-behavior).

# Container lifecycle hooks

Since Modal will reuse the same container for multiple inputs, sometimes you
might want to run some code exactly once when the container starts or exits.

To accomplish this, you need to use Modal's class syntax and the
[`@app.cls`](/docs/reference/modal.App#cls) decorator. Specifically, you'll
need to:

1. Convert your function to a method by making it a member of a class.
2. Decorate the class with `@app.cls(...)` with same arguments you previously
   had for `@app.function(...)`.
3. Instead of the `@app.function` decorator on the original method, use
   `@modal.method` or the appropriate decorator for a
   [web endpoint](#lifecycle-hooks-for-web-endpoints).
4. Add the correct method "hooks" to your class based on your need:
   - `@modal.enter` for one-time initialization (remote)
   - `@modal.exit` for one-time cleanup (remote)

## `@modal.enter`

The container entry handler is called when a new container is started. This is
useful for doing one-time initialization, such as loading model weights or
importing packages that are only present in that image.

To use, make your function a member of a class, and apply the `@modal.enter()`
decorator to one or more class methods:

```python
import modal

app = modal.App()

@app.cls(cpu=8)
class Model:
    @modal.enter()
    def run_this_on_container_startup(self):
        import pickle
        self.model = pickle.load(open("model.pickle"))

    @modal.method()
    def predict(self, x):
        return self.model.predict(x)


@app.local_entrypoint()
def main():
    Model().predict.remote(x=123)
```

When working with an [asynchronous Modal](/docs/guide/async) app, you may use an
async method instead:

```python
import modal

app = modal.App()

@app.cls(memory=1024)
class Processor:
    @modal.enter()
    async def my_enter_method(self):
        self.cache = await load_cache()

    @modal.method()
    async def run(self, x):
        return await do_some_async_stuff(x, self.cache)


@app.local_entrypoint()
async def main():
    await Processor().run.remote(x=123)
```

Note: The `@modal.enter()` decorator replaces the earlier `__enter__` syntax, which
has been deprecated.

## `@modal.exit`

The container exit handler is called when a container is about to exit. It is
useful for doing one-time cleanup, such as closing a database connection or
saving intermediate results. To use, make your function a member of a class, and
apply the `@modal.exit()` decorator:

```python
import modal

app = modal.App()

@app.cls()
class ETLPipeline:
    @modal.enter()
    def open_connection(self):
        import psycopg2
        self.connection = psycopg2.connect(os.environ["DATABASE_URI"])

    @modal.method()
    def run(self):
        # Run some queries
        pass

    @modal.exit()
    def close_connection(self):
        self.connection.close()


@app.local_entrypoint()
def main():
    ETLPipeline().run.remote()
```

Exit handlers are also called when a container is [preempted](/docs/guide/preemption).
The exit handler is given a grace period of 30 seconds to finish, and it will be
killed if it takes longer than that to complete.

## Lifecycle hooks for web endpoints

Modal `@function`s that are [web endpoints](/docs/guide/webhooks) can be
converted to the class syntax as well. Instead of `@modal.method`, simply use
whichever of the web endpoint decorators (`@modal.fastapi_endpoint`,
`@modal.asgi_app` or `@modal.wsgi_app`) you were using before.

```python
from fastapi import Request

import modal

image = modal.Image.debian_slim().pip_install("fastapi")
app = modal.App("web-endpoint-cls", image=image)

@app.cls()
class Model:
    @modal.enter()
    def run_this_on_container_startup(self):
        self.model = pickle.load(open("model.pickle"))

    @modal.fastapi_endpoint()
    def predict(self, request: Request):
        ...
```
