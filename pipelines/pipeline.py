from zenml import pipeline
from steps.training_step import training_step


@pipeline
def snake_pipeline(resume: bool = True):
    training_step(
        cell_number=25,
        cell_size=30,
        rendering=False,
        num_episodes=10_000,
        model_path="models/dqn_model.pth",
        buffer_path="models/replay_buffer.pkl",
        step_counter_path="models/step_counter.pkl",
        resume=resume
    )
