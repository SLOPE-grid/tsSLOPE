
using PyCall

repo_dir = dirname(Base.source_path())
ENV["PYTHON"] = string(repo_dir,"/.pyenv/bin/python3")

pushfirst!(pyimport("sys")."path", repo_dir)

tsslope_lib = pyimport("tsslope_lib")

jl_lib = string(repo_dir,"/tsslope-pump-jl")

include(string(jl_lib,"/tsi_constraints.jl"))



