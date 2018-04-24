#================================
# [Usage]
# For each step, designate followings
#   src_*: Used source files
#   dep_*: Precursor steps
#   out_*: Output files for "make clean"
#   cmd_*: Command to be executed
#================================

STEP_ALL = prep ETKF hybrid

src_prep = generate_observations.py generate_nature_run.py runall_tutorial_1.sh
dep_prep =
out_prep = x_nature.pkl x_freerun.pkl y_obs.pkl
cmd_prep = sh runall_tutorial_1.sh

src_ETKF = runall_tutorial_3.sh analysis_init.py class_da_system.py \
           generate_analysis_3dEns.py
dep_ETKF = prep
out_ETKF = x_analysis_init.pkl Pb_hist.npy x_analysis_ETKF.pkl ETKF_vs_nature.html
cmd_ETKF = sh runall_tutorial_3.sh ETKF

src_hybrid = runall_tutorial_3.sh analysis_init.py class_da_system.py \
             generate_analysis_3dEns.py
dep_hybrid = prep
out_hybrid = x_analysis_init.pkl Pb_hist.npy x_analysis_hybrid.pkl hybrid_vs_nature.html
cmd_hybrid = sh runall_tutorial_3.sh hybrid

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

