import sys
import subprocess

proj_name = "gnn_ext"

site_paths = sys.path
site_path = list(filter(lambda x: x.split("/")[-1] == "site-packages", site_paths))[0]

ext_suffix = subprocess.check_output(["python3-config", "--extension-suffix"]).decode()[:-1]

ext_path = "build/" + proj_name + ext_suffix

print("MESSAGE:", "ext_path:", ext_path, "site_path:", site_path)

subprocess.check_call(["cp", ext_path, site_path])