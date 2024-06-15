from train import train_dqn
from utils import render_video_of_model, show_video

if __name__ == "__main__":
    agent = train_dqn()
    render_video_of_model(agent, 'LunarLander-v2')
    show_video()
