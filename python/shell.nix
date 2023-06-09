let
  pkgs = import <nixpkgs> {};
  CPLEX20="/home/sat/cplex20";
  PYTHONVERSION="3.8";
  ARCH="x86-64_linux";
  PYTHONHOME="/home/sat/python-packages";

in pkgs.mkShell {
  buildInputs = [
    pkgs.tree
    pkgs.python3
    pkgs.python3.pkgs.pip
    pkgs.python3.pkgs.tqdm
    #pkgs.python3.pkgs.pandas
    pkgs.python3.pkgs.pytorch
    #pkgs.python3.pkgs.gym
    pkgs.python3.pkgs.tensorflow-tensorboard
    pkgs.python3.pkgs.matplotlib
    #pkgs.python3.pkgs.sklearn
  ];
  shellHook = ''
	# Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
	# See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
	export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/libstdc++.so.6
	export PIP_PREFIX=$(pwd)/_build/pip_packages
	export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
	export PATH="$PIP_PREFIX/bin:$PATH"
    
    #export CPLEX20=/home/sat/cplex2210
    #export CPLEX20=/home/sat/cplex2210_python/lib/python/cplex
	unset SOURCE_DATE_EPOCH

	export PYTHONPATH=$PYTHONPATH:${CPLEX20}
	#export PYTHONPATH=$PYTHONPATH:${CPLEX20}/cplex/python/${PYTHONVERSION}/${ARCH}:${PYTHONHOME}/lib/python/cplex




  '';
}

