import re
import matplotlib.pyplot as plt
import os
import argparse

drop_rows = []
drop_iters = []


def read_logs(log_file):
    losses = {"iter": [], "loss": [], "time": []}
    ckpts_time = []
    with open(log_file) as f:
        for log_line in f:
            iteration_match = re.search(r"iteration\s+(\d+)/\s*\d+", log_line)
            lm_loss_match = re.search(
                r"lm loss:\s*([-+]?\d*\.\d+([eE][-+]?\d+)?)", log_line
            )
            time_match = re.search(
                r"elapsed time per iteration \(ms\):\s*([-+]?\d*\.\d+)", log_line
            )

            if iteration_match and lm_loss_match:
                iteration = iteration_match.group(1)
                lm_loss = lm_loss_match.group(1)
                time_elapsed = time_match.group(1)
                losses["iter"].append(int(iteration))
                losses["loss"].append(float(lm_loss))
                losses["time"].append(float(time_elapsed))
            elif iteration_match:
                iteration = iteration_match.group(1)
                print(f"No loss found for iteration {iteration}. Skipping.")
                drop_iters.append(int(iteration))

            save_ckpt_match = re.search(
                r"save-checkpoint\s*\.+\s*:\s*\(([-+]?\d*\.\d+),\s*([-+]?\d*\.\d+)\)",
                log_line,
            )
            if save_ckpt_match:
                ckpt_time = save_ckpt_match.group(1)
                ckpts_time.append(float(ckpt_time))

    return losses, ckpts_time


def read_elastic_log(log_file):
    """Read loss data from ElasticMegatron log file (one loss value per line)."""
    losses = {"iter": [], "loss": [], "time": []}
    with open(log_file) as f:
        for i, log_line in enumerate(f, 1):
            log_line = log_line.strip()
            if not log_line or "transfer at" in log_line:
                continue
            try:
                loss_val = float(log_line)
                losses["iter"].append(i)
                losses["loss"].append(loss_val)
                # No time data in this format, set to 0
                losses["time"].append(0)
            except ValueError:
                print(f"Could not parse line: {log_line}")
    return losses, []


def is_elastic_log(log_file):
    """Determine if the log file is from ElasticMegatron (just loss values)."""
    with open(log_file) as f:
        lines_to_check = 5
        lines_checked = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            lines_checked += 1
            if any(c.isalpha() for c in line):
                return False
            try:
                float(line)
            except ValueError:
                return False

            if lines_checked >= lines_to_check:
                break

    return lines_checked > 0


def _friendly_label(file_name):
    norm = os.path.normpath(file_name)
    parts = norm.split(os.sep)
    try:
        if len(parts) >= 3 and parts[-1].lower() == "loss.txt":
            parent = parts[-2]
            if parent == "baseline" and len(parts) >= 4:
                case = parts[-3]
                return f"{case} (baseline)"
            else:
                return parent
    except Exception:
        pass

    # Fallbacks
    base = os.path.basename(file_name)
    if "baseline" in parts:
        return f"{base} (baseline)"
    return base


def plot_curves(log_files, out_dir=None):
    loss_data = []
    for log_file in log_files:
        print(f"reading log file {log_file}")
        if is_elastic_log(log_file):
            print(f"Detected ElasticMegatron loss log format for {log_file}")
            losses, _ = read_elastic_log(log_file)
        else:
            print(f"Detected standard log format for {log_file}")
            losses, _ = read_logs(log_file)
        loss_data.append(losses)

    # loss curve
    ax = plt.subplot()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    for one_loss, file_name in zip(loss_data, log_files):
        file_display_name = _friendly_label(file_name)
        if len(file_display_name) > 50:
            file_display_name = f"{file_display_name[:25]}…{file_display_name[-24:]}"
        plt.plot(one_loss["iter"], one_loss["loss"], label=file_display_name)
    plt.legend()
    save_path = (
        os.path.join(out_dir, "loss_curves.png") if out_dir else "loss_curves.png"
    )
    ax.figure.savefig(save_path)
    plt.title("Loss Curves")

    plt.clf()

    # Only plot time curves if there's actual time data
    has_time_data = False
    for one_loss in loss_data:
        if any(one_loss["time"]):
            has_time_data = True
            break

    if has_time_data:
        # time elapsed curve
        bx = plt.subplot()
        bx.set_xlabel("Iteration")
        bx.set_ylabel("Elapsed time (ms)")
        for one_loss, file_name in zip(loss_data, log_files):
            if not any(one_loss["time"]):
                continue  # Skip files with no time data
            file_display_name = _friendly_label(file_name)
            if len(file_display_name) > 50:
                file_display_name = (
                    f"{file_display_name[:25]}…{file_display_name[-24:]}"
                )
            if len(one_loss["time"]) > 1:
                avg_time = sum(one_loss["time"][1:]) / len(one_loss["time"][1:])
                print(f"{file_display_name} avg time: {avg_time}ms")
            plt.plot(one_loss["iter"], one_loss["time"], label=file_display_name)
        plt.legend()
        plt.title("Time Curves")
        save_path = (
            os.path.join(out_dir, "time_curves.png") if out_dir else "time_curves.png"
        )
        bx.figure.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curves from log/loss files")
    parser.add_argument(
        "files", nargs="+", help="List of loss/log files to plot together"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to CWD)",
    )
    args = parser.parse_args()

    plot_curves(args.files, out_dir=args.out_dir)
