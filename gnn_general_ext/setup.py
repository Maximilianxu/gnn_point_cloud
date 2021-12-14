# NOTE: deprecated: this will generate errors
print("use build.sh instead")
exit()

from distutils.core import setup
from distutils.cmd import Command
import subprocess

proj_name = "gnn_ext"
version = "0.0.1"

class InstallProj(Command):
  user_options = []
  
  def __init__(self, dist) -> None:
    super().__init__(dist)

  def initialize_options(self):
      pass

  # This method must be implemented
  def finalize_options(self):
      pass

  def run(self):
    import sys
    site_paths = sys.path
    site_path = list(filter(lambda x: x.split("/")[-1] == "site-packages", site_paths))[0]

    ext_suffix = subprocess.check_output(["python3-config", "--extension-suffix"]).decode()[:-1]

    ext_path = proj_name + ext_suffix

    print("MESSAGE:", "ext_path:", ext_path, "site_path:", site_path)

    subprocess.check_call(["cp", ext_path, site_path])

setup(
  name=proj_name,
  version=version,
  author="zyx",
  author_email="don\' tell you",
  cmdclass={"install": InstallProj}
)