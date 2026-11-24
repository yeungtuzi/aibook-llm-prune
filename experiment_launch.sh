#!/usr/bin/env bash
# Deploy exact local Git HEAD and start one private AIBOOK smoke unit. No implicit stop.
set -euo pipefail
mode=${1:?mode}; slots=${2:?slots}; seconds=${3:-1500}; [[ $slots == 128 || $slots == 140 || $slots == 144 || $slots == 8 || $slots == 160 ]]; [[ $seconds =~ ^[0-9]+$ && $seconds -le 2400 ]]
knob2=
knob3=
knob4=
knob5=
case $mode in eager) knob=AIBOOK_MODULE_TIMING=0;; routedgraph) knob=AIBOOK_ROUTED_GRAPH=1;; mapped) knob=AIBOOK_MAPPED_SLOTS=1;; projection) knob=AIBOOK_PROJECTION_GRAPH=1;; graphprobe) knob=AIBOOK_GRAPH_PROBE=1; knob2=AIBOOK_EAGER=0;; graphtiming) knob=AIBOOK_GRAPH_PROBE=1; knob2=AIBOOK_EAGER=0; knob3=AIBOOK_SLOT_TIMING=1;; resident) knob=AIBOOK_RESIDENT=1; knob2=AIBOOK_EAGER=0;; residentmtp) knob=AIBOOK_RESIDENT=1; knob2=AIBOOK_EAGER=0; knob3=AIBOOK_MTP=1;; graphexpert) knob=AIBOOK_GRAPH_PROBE=1; knob2=AIBOOK_EAGER=0; knob3=AIBOOK_EXPERT_GRAPH=1;; graphoverlap) knob=AIBOOK_GRAPH_PROBE=1; knob2=AIBOOK_EAGER=0; knob3=AIBOOK_EXPERT_GRAPH=1; knob4=AIBOOK_OVERLAP=1;; graphonefill) knob=AIBOOK_GRAPH_PROBE=1; knob2=AIBOOK_EAGER=0; knob3=AIBOOK_EXPERT_GRAPH=1; knob4=AIBOOK_OVERLAP=1; knob5=AIBOOK_FILL_ONE=1;; graphexperttiming) knob=AIBOOK_GRAPH_PROBE=1; knob2=AIBOOK_EAGER=0; knob3=AIBOOK_EXPERT_GRAPH=1; knob4=AIBOOK_SLOT_TIMING=1;; fitted) knob=AIBOOK_LAYER_BUDGET=/home/lf/sglang-m1000/trial-$(git rev-parse --short HEAD)-$slots-fitted/measurements/layer-budget-heldout.json;; *) exit 2;; esac
root=$(git rev-parse --show-toplevel); rev=$(git rev-parse --short HEAD); [[ $(pwd -P) == "$root" ]]; host=lf@192.168.100.193; key=$HOME/.ssh/id_ed25519_aibook
run=trial-$rev-$slots-$mode; d=sglang-m1000/$run; unit=aibook-slot-smoke-$run
ssh -i "$key" "$host" "set -e; test -z \"\$(ss -Hltn 'sport = :30141')\"; test -z \"\$(systemctl --user list-units --state=running --no-legend 'aibook-slot-smoke-*.service')\"; mkdir $d"
git archive HEAD integration guard_launch.py stream_bench.py trace measurements/layer-budget-heldout.json | ssh -i "$key" "$host" "tar -xm -C $d"
ssh -i "$key" "$host" "set -e; cd $d; python3 integration/overlay.py ~/miniforge3/envs/sglang-py310/lib/python3.10/site-packages --out overlay; systemd-run --user --unit=$unit -p MemoryMax=18000000000 -p MemorySwapMax=0 --setenv=AIBOOK_SLOTS=$slots --setenv=$knob ${knob2:+--setenv=$knob2} ${knob3:+--setenv=$knob3} ${knob4:+--setenv=$knob4} ${knob5:+--setenv=$knob5} ${AIBOOK_CONTEXT:+--setenv=AIBOOK_CONTEXT=$AIBOOK_CONTEXT} ${AIBOOK_MAX_TOKENS:+--setenv=AIBOOK_MAX_TOKENS=$AIBOOK_MAX_TOKENS} --setenv=AIBOOK_OFFLINE_SMOKE=YES --setenv=AIBOOK_OVERLAY=\$PWD/overlay /usr/bin/python3 -u \$PWD/guard_launch.py --seconds $seconds --cap-gb ${AIBOOK_GUARD_CAP_GB:-24} --margin-gb ${AIBOOK_GUARD_MARGIN_GB:-2} /bin/bash \$PWD/integration/smoke.sh"
