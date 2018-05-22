#================================
# [Usage]
# For each step, designate followings
#   src_step: Used source files
#   dep_step: Precursor steps
#   out_step: Output files for "make clean"
#   cmd_step: Command to be executed
#================================

STEP_ALL = plot copy

src_plot := module_plot.py
dep_plot :=
out_plot := img/*.pdf
cmd_plot := mkdir -p img && python module_plot.py

src_copy :=
dep_copy := plot
out_copy := /lustre/tyoshida/shrt/sync/maooam/img
cmd_copy := rm -rf /lustre/tyoshida/shrt/sync/maooam/img && cp -r img /lustre/tyoshida/shrt/sync/maooam

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

