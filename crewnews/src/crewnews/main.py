#!/usr/bin/env python
import sys
import warnings
import time
import phoenix as px
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.crewai import CrewAIInstrumentor

from datetime import datetime

from crewnews.crew import Crewnews

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }
    
    session = px.launch_app()
    
    print(f"Phoenix Session URL: {session.url}")
    
    tracer_provider = trace_sdk.TracerProvider()
    span_exporter = OTLPSpanExporter(f"{session.url}v1/traces")
    span_processor = SimpleSpanProcessor(span_exporter)
    tracer_provider.add_span_processor(span_processor)
    trace_api.set_tracer_provider(tracer_provider)

    CrewAIInstrumentor().instrument(skip_dep_check=True)
    
    try:
        Crewnews().crew().kickoff(inputs=inputs)
        pass
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
    
    # Keep the Phoenix session running by adding a blocking loop
    print("Phoenix session is running... Press Ctrl+C to stop.")
    
    while True:
        time.sleep(60)  # Keep the process alive


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Crewnews().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Crewnews().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Crewnews().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
