#!/usr/bin/env bash
# Prints a 4-line PASS/FAIL summary for the resume test.
cd /home/nikhil/source_code/RAPTOR
A=logs/resume_testA.log; B=logs/resume_testB.log
p(){ [ "$1" = ok ] && echo "PASS - $2" || echo "FAIL - $2"; }

grep -q "unfroze last" "$A" 2>/dev/null && p ok "1 unfreeze fired in phase A" || p no "1 unfreeze fired in phase A"
grep -q "resume optimizer-group reconcile" "$B" 2>/dev/null && p ok "2 resume patch ran" || p no "2 resume patch ran"
grep -qiE "different number of parameter groups|Traceback|ValueError" "$B" 2>/dev/null && p no "3 no param-group crash" || p ok "3 no param-group crash"
grep -q "Training completed" "$B" 2>/dev/null && p ok "4 phase B resumed + finished" || p no "4 phase B resumed + finished"
