#================================
# [Usage]
# For each step, designate followings
#   src_*: Used source files
#   dep_*: Precursor steps
#   out_*: Output files for "make clean"
#   cmd_*: Command to be executed
#================================

STEP_ALL = compile prep hybrid plot tex

src_compile =
dep_compile =
out_compile = step_maooam.so
cmd_compile = cd maooam_fortran && make clean && make && cd ../ && cp maooam_fortran/step_maooam.so .

src_prep = generate_observations.py generate_nature_run.py runall_tutorial_1.sh \
           module_obs_network.py params_maooam.py tl_ad.py
dep_prep = compile
out_prep = x_nature.pkl x_freerun.pkl y_obs.pkl
cmd_prep = sh runall_tutorial_1.sh

src_nudging = runall_tutorial_2.sh analysis_init.py class_da_system.py module_constants.py \
              generate_analysis_3dDet.py
dep_nudging = prep
out_nudging = x_analysis_init.pkl
cmd_nudging = sh runall_tutorial_2.sh nudging

src_3DVar = runall_tutorial_2.sh analysis_init.py class_da_system.py module_constants.py \
              generate_analysis_3dDet.py
dep_3DVar = prep
out_3DVar = x_analysis_init.pkl
cmd_3DVar = sh runall_tutorial_2.sh 3DVar

src_ETKF = runall_tutorial_3.sh analysis_init.py class_da_system.py module_constants.py \
           generate_analysis_3dEns.py
dep_ETKF = prep
out_ETKF = x_analysis_init.pkl Pb_hist.npy x_analysis_ETKF.pkl ETKF_vs_nature.html
cmd_ETKF = sh runall_tutorial_3.sh ETKF

src_hybrid = runall_tutorial_3.sh analysis_init.py class_da_system.py module_constants.py \
           generate_analysis_3dEns.py
dep_hybrid = prep
out_hybrid = x_analysis_init.pkl Pb_hist.npy x_analysis_hybrid.pkl hybrid_vs_nature.html
cmd_hybrid = sh runall_tutorial_3.sh hybrid

src_plot = module_plot.py plot_error.py
dep_plot = hybrid
out_plot = img rmse_*.npy
cmd_plot = mkdir -p img/hybrid && python module_plot.py && python plot_error.py hybrid
# cmd_plot = mkdir -p img/3DVar img/ETKF img/hybrid && python module_plot.py && python plot_error.py 3DVar && python plot_error.py ETKF && python plot_error.py hybrid

src_tex =
dep_tex = plot
out_tex = out.pdf
cmd_tex = ~/repos/works/2018/dir_to_latex/main.py img 9

# ===============================================
# end of settings
# ===============================================

VPATH = .make
all: $(STEP_ALL)

define rule_comm
$1: $(src_$1) $(dep_$1) Makefile
	@echo ""
	@echo "STEP $1:"
	$(cmd_$1)
	@mkdir -p $(VPATH)
	@touch $(VPATH)/$1
clean_$1:
	@echo "clean $1:"
	rm -rf $(out_$1)
	@rm -f $(VPATH)/$1
endef

$(foreach i, $(STEP_ALL), $(eval $(call rule_comm,$i)))

clean:
	rm -rf $(foreach i,$(STEP_ALL),$(out_$i))
	@rm -rf $(VPATH)

.PHONY: all clean

