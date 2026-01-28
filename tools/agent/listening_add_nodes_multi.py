import socket
import subprocess

# from megatron.training.training import get_parallel_strategy_list
import os


# new node keeps listening for the command
def listen_for_master_command():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 23455))
        s.listen()
        print("Listening for master node...")
        s.accept()
        # tp = get_parallel_strategy_list().get("tensor_model_parallel_size", 1)
        # pp = get_parallel_strategy_list().get("pipeline_model_parallel_size", 1)
        pp = "2"
        tp = "4"
        world_size = "2"

        env = os.environ.copy()
        env["TP"] = str(tp)
        env["PP"] = str(pp)
        env["NNODES"] = str(world_size)
        env["RANK"] = "1"
        env["MASTER_PORT"] = "6369"
        env["NEW_NODE"] = "1"

        cmd = ["../../run_e2e_demo.sh"]
        subprocess.Popen(cmd, env=env)


if __name__ == "__main__":
    listen_for_master_command()
