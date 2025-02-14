
#import Pkg


#Pkg.add("PyCall")
#Pkg.build("PyCall")

using PyCall

repo_dir = dirname(Base.source_path())
ENV["PYTHON"] = string(repo_dir,"/.pyenv/bin/python3")

#println(ENV["PYTHON"])

#exit()
#Pkg.add("PyCall")
#Pkg.build("PyCall")

#using PyCall

pushfirst!(pyimport("sys")."path", repo_dir)

#println(repo_dir)
#println(ENV["PYTHON"])
#exit()
tsslope_lib = pyimport("tsslope_lib")
#exit()

jl_lib = string(repo_dir,"/tsslope_lib_jl")

include(string(jl_lib,"/tsi_constraints.jl"))



