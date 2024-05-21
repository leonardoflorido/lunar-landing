from train import train_dqn
from utils import show_video_of_model, show_video

if __name__ == "__main__":
    train_dqn()
    show_video_of_model('LunarLander-v2')
    show_video()
