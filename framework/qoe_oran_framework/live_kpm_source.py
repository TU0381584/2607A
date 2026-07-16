"""Live KpmSource: the real, RIC-free E2 loop, talking directly to OAI's
gNB via its built-in `E2_AGENT` UDP control loop
(oai_ran/openair2/E2_AGENT/e2_agent_app.c, unconditionally started at gNB
boot -- NOT the CMake-gated FlexRIC E2_AGENT, a different, dormant thing
of the same name; NOT the o-ran-e2sim/kpm_sim pipeline, which was found to
require a real E2AP/SCTP RIC Subscription Request before it relays any
data and is therefore not usable without a RIC).

Wire protocol, confirmed by reading e2_agent_app.c/e2_message_handlers.c:
  - gNB listens on UDP 0.0.0.0:6655 (E2AGENT_IN_PORT) for
    INDICATION_REQUEST and CONTROL RANMessage protobufs.
  - gNB sends responses to UDP 127.0.0.1:6600 (E2AGENT_OUT_PORT).
  - This is request/response, not a subscribe-and-stream model: each
    INDICATION_REQUEST gets exactly one INDICATION_RESPONSE back; CONTROL
    messages get no response at all (fire-and-forget, applied directly to
    the live gNB_MAC_INST slicing policy via apply_slicing_ctrl()).
  - CRITICAL: SUBSCRIPTION-type messages are unimplemented and crash the
    gNB process (assert(0!=0) in handle_subscription()) -- never send one.
    This module only ever builds INDICATION_REQUEST and CONTROL messages.

This corrects an earlier version of this module, which targeted the
xapp-oai/base-xapp UDP-7001/TCP-4200 transport (the o-ran-e2sim/kpm_sim
pipeline) before the RIC dependency above was discovered. That transport
is kept nowhere in this file now; this one is UDP-only end to end,
matching xapp-oai/base-xapp/xapp_control_ricbypass.py's port pairing
(in_port=6600, out_port=6655) -- though that module itself is unused here
(a thin demo wrapper, not an importable library); the send/receive calls
below are plain socket calls, reimplemented directly.

Environment note, discovered empirically: the pre-generated
ran_messages_pb2.py raises "Descriptors cannot be created directly" under
this environment's protobuf 4.25.6 (C++-backed) implementation. Forcing
the pure-Python implementation via PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION
fixes this without touching the xapp-oai repo or downgrading protobuf.
"""

import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

DEFAULT_PROTO_BUILD_DIR = "/home/w5/ors/xapp-oai/base-xapp/oai-oran-protolib/builds"
_proto_dir = os.environ.get("XAPP_OAI_PROTO_DIR", DEFAULT_PROTO_BUILD_DIR)
if not Path(_proto_dir).is_dir():
    raise ImportError(
        f"protobuf build dir not found at {_proto_dir!r}. Set XAPP_OAI_PROTO_DIR "
        "to a directory containing ran_messages_pb2.py (e.g. "
        "<xapp-oai>/base-xapp/oai-oran-protolib/builds, or "
        "<oai_ran>/openair2/E2_AGENT/oai-oran-protolib/builds -- both ship an "
        "identical generated module)."
    )
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

import ran_messages_pb2  # noqa: E402

from .types import UeSample  # noqa: E402

# gNB's ports, from oai_ran/openair2/E2_AGENT/e2_agent_app.h -- named from
# the gNB's own perspective there (E2AGENT_IN_PORT/E2AGENT_OUT_PORT); this
# module names them from the xApp's perspective instead (the port the
# xApp SENDS TO is the gNB's IN port, and vice versa).
DEFAULT_HOST = "127.0.0.1"
GNB_LISTEN_PORT = 6655   # xApp sends INDICATION_REQUEST/CONTROL here
XAPP_LISTEN_PORT = 6600  # xApp receives INDICATION_RESPONSE here
MAX_RECV_BYTES = 4096


def build_indication_request() -> bytes:
    message = ran_messages_pb2.RAN_message()
    message.msg_type = ran_messages_pb2.RAN_message_type.INDICATION_REQUEST
    inner = ran_messages_pb2.RAN_indication_request()
    inner.target_params.extend(
        [ran_messages_pb2.RAN_parameter.GNB_ID, ran_messages_pb2.RAN_parameter.UE_LIST]
    )
    message.ran_indication_request.CopyFrom(inner)
    return message.SerializeToString()


def build_control_request(sst: int, sd: int, min_ratio: int, max_ratio: int) -> bytes:
    slicing = ran_messages_pb2.slicing_control_m()
    slicing.sst = int(sst)
    slicing.sd = int(sd)
    slicing.min_ratio = int(min_ratio)
    slicing.max_ratio = int(max_ratio)

    entry = ran_messages_pb2.RAN_param_map_entry()
    entry.key = ran_messages_pb2.RAN_parameter.SLICING_CONTROL
    entry.slicing_ctrl.CopyFrom(slicing)

    inner = ran_messages_pb2.RAN_control_request()
    inner.target_param_map.append(entry)

    message = ran_messages_pb2.RAN_message()
    message.msg_type = ran_messages_pb2.RAN_message_type.CONTROL
    message.ran_control_request.CopyFrom(inner)
    return message.SerializeToString()


def parse_indication_response(data: bytes, gnb_id: str, timestamp_s: float) -> List[UeSample]:
    response = ran_messages_pb2.RAN_indication_response()
    response.ParseFromString(data)
    samples: List[UeSample] = []
    for entry in response.param_map:
        if entry.key != ran_messages_pb2.RAN_parameter.UE_LIST:
            continue
        for ue in entry.ue_list.ue_info:
            samples.append(
                UeSample(
                    rnti=ue.rnti,
                    timestamp_s=timestamp_s,
                    nssai_sst=ue.nssai_sST,
                    nssai_sd=ue.nssai_sD,
                    avg_prbs_dl=ue.avg_prbs_dl,
                    gnb_id=gnb_id,
                    dl_total_bytes=ue.dl_total_bytes,
                    dl_errors=ue.dl_errors,
                    dl_bler=ue.dl_bler,
                    dl_mac_buffer_occupation=ue.dl_mac_buffer_occupation,
                )
            )
    return samples


class LiveKpmSource:
    """Real E2-loop transport implementing the KpmSource protocol, over
    OAI's built-in RIC-free UDP E2_AGENT (see module docstring).

    Request/response, not subscribe-and-stream: poll() sends a fresh
    INDICATION_REQUEST and blocks for exactly one INDICATION_RESPONSE, so
    the live cadence is caller-driven (RANEnv.step()'s loop), not a
    background report timer.
    """

    def __init__(
        self,
        gnb_id: str,
        host: str = DEFAULT_HOST,
        xapp_listen_port: int = XAPP_LISTEN_PORT,
        gnb_listen_port: int = GNB_LISTEN_PORT,
        recv_timeout_s: float = 30.0,
    ):
        self._gnb_id = gnb_id
        self._host = host
        self._gnb_listen_port = gnb_listen_port
        self._recv_timeout_s = recv_timeout_s

        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._recv_sock.bind(("", xapp_listen_port))
        self._recv_sock.settimeout(recv_timeout_s)

        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.sent_controls: List[Dict[str, Any]] = []

    def poll(self) -> List[UeSample]:
        buf = build_indication_request()
        self._send_sock.sendto(buf, (self._host, self._gnb_listen_port))
        try:
            data, _addr = self._recv_sock.recvfrom(MAX_RECV_BYTES)
        except socket.timeout:
            raise TimeoutError(
                f"no INDICATION_RESPONSE from gNB E2 agent within {self._recv_timeout_s}s "
                f"(sent request to {self._host}:{self._gnb_listen_port}, listening on "
                f"port {self._recv_sock.getsockname()[1]}) -- is nr-softmodem running with "
                "the ricbypass E2_AGENT built in (native oai_ran build, not the oaignb.yaml "
                "Docker service, which clones vanilla upstream OAI without this code)?"
            )
        return parse_indication_response(data, self._gnb_id, time.time())

    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None:
        buf = build_control_request(sst, sd, min_ratio, max_ratio)
        self._send_sock.sendto(buf, (self._host, self._gnb_listen_port))
        self.sent_controls.append(
            {"gnb_id": gnb_id, "sst": sst, "sd": sd, "min_ratio": min_ratio, "max_ratio": max_ratio}
        )

    def notify_rejected(self, gnb_id: str, slice_id: str, n_rejected: int) -> None:
        # No-op on the real rig: a rejected request's actual effect on the
        # real scheduler's backlog is whatever the next poll() reports back
        # -- there's no separate "relieve this much backlog" signal to send
        # over E2 (only slicing_control_m's ratio ceiling exists), unlike
        # ClosedLoopKpmSource's synthetic model, which needs an explicit
        # notification because its offered/backlog state is otherwise
        # entirely decoupled from admission decisions.
        pass

    def close(self) -> None:
        self._recv_sock.close()
        self._send_sock.close()
