import torch
import random
import pickle

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(action).to(self.device),
            torch.tensor(reward).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor(done).to(self.device),
        )

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.stack(state).to(self.device),
            torch.stack(action).to(self.device),
            torch.stack(reward).to(self.device),
            torch.stack(next_state).to(self.device),
            torch.stack(done).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)
    
    def save_to_file(self, file_path):
        buffer_dict = {
            "observations": [item[0].cpu().numpy() for item in self.buffer],
            "actions": [item[1].cpu().numpy() for item in self.buffer],
            "rewards": [item[2].cpu().numpy() for item in self.buffer],
            "next_states": [item[3].cpu().numpy() for item in self.buffer],
            "dones": [item[4].cpu().numpy() for item in self.buffer],
        }
        with open(file_path, "wb") as f:
            pickle.dump(buffer_dict, f)