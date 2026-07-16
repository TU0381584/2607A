"""End-to-end transport tests against a real loopback UDP pair (a fake
gNB E2_AGENT peer run in a background thread) -- no real gNB needed to
verify the wire protocol itself is correct."""

import os
import socket
import sys
import threading
from pathlib import Path

import pytest

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_proto_dir = os.environ.get(
    "XAPP_OAI_PROTO_DIR",
    "/home/w5/ors/xapp-oai/base-xapp/oai-oran-protolib/builds",
)
if not Path(_proto_dir).is_dir():
    pytest.skip(f"xapp-oai protobuf build dir not present at {_proto_dir}", allow_module_level=True)

from qoe_oran_framework.live_kpm_source import (  # noqa: E402
    LiveKpmSource,
    build_control_request,
    build_indication_request,
    parse_indication_response,
)

if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)
import ran_messages_pb2  # noqa: E402

XAPP_LISTEN_PORT = 26600
GNB_LISTEN_PORT = 26655


def test_build_indication_request_shape():
    buf = build_indication_request()
    msg = ran_messages_pb2.RAN_message()
    msg.ParseFromString(buf)
    assert msg.msg_type == ran_messages_pb2.RAN_message_type.INDICATION_REQUEST
    assert list(msg.ran_indication_request.target_params) == [
        ran_messages_pb2.RAN_parameter.GNB_ID, ran_messages_pb2.RAN_parameter.UE_LIST,
    ]


def test_build_control_request_round_trips_fields():
    buf = build_control_request(sst=1, sd=1, min_ratio=5, max_ratio=30)
    msg = ran_messages_pb2.RAN_message()
    msg.ParseFromString(buf)
    slicing = msg.ran_control_request.target_param_map[0].slicing_ctrl
    assert (slicing.sst, slicing.sd, slicing.min_ratio, slicing.max_ratio) == (1, 1, 5, 30)


def test_never_builds_a_subscription_message():
    """Sending a SUBSCRIPTION-type message crashes the real gNB
    (handle_subscription() is assert(0!=0) in e2_message_handlers.c) --
    this module must only ever emit INDICATION_REQUEST or CONTROL."""
    for buf in (build_indication_request(), build_control_request(1, 1, 5, 30)):
        msg = ran_messages_pb2.RAN_message()
        msg.ParseFromString(buf)
        assert msg.msg_type != ran_messages_pb2.RAN_message_type.SUBSCRIPTION


def test_parse_indication_response_extracts_ue_samples():
    resp = ran_messages_pb2.RAN_indication_response()
    entry = resp.param_map.add()
    entry.key = ran_messages_pb2.RAN_parameter.UE_LIST
    entry.ue_list.connected_ues = 1
    ue = entry.ue_list.ue_info.add()
    ue.rnti = 99
    ue.avg_prbs_dl = 12.5
    ue.nssai_sST = 1
    ue.nssai_sD = 1
    ue.dl_mac_buffer_occupation = 3.0
    ue.dl_bler = 0.1
    ue.dl_errors = 0.2

    samples = parse_indication_response(resp.SerializeToString(), gnb_id="gnb-0", timestamp_s=1.0)
    assert len(samples) == 1
    assert samples[0].rnti == 99
    assert samples[0].avg_prbs_dl == pytest.approx(12.5)
    assert samples[0].gnb_id == "gnb-0"
    assert samples[0].nssai_sd == 1


def _fake_gnb_agent(ready, received_first_request, stop_event, received_control):
    """Simulates OAI's E2_AGENT: binds the gNB's listen port, replies to
    the first INDICATION_REQUEST with one fabricated UE, then captures the
    next datagram (expected to be a CONTROL message) and stops -- mirrors
    e2_agent_app.c's request/response (not subscribe/stream) behaviour."""
    gnb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gnb_sock.bind(("127.0.0.1", GNB_LISTEN_PORT))
    gnb_sock.settimeout(5.0)
    ready.set()

    reply_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    data, _addr = gnb_sock.recvfrom(4096)
    received_first_request["data"] = data

    resp = ran_messages_pb2.RAN_indication_response()
    entry = resp.param_map.add()
    entry.key = ran_messages_pb2.RAN_parameter.UE_LIST
    entry.ue_list.connected_ues = 1
    ue = entry.ue_list.ue_info.add()
    ue.rnti = 7
    ue.avg_prbs_dl = 4.0
    ue.nssai_sST = 1
    ue.nssai_sD = 1
    reply_sock.sendto(resp.SerializeToString(), ("127.0.0.1", XAPP_LISTEN_PORT))

    control_data, _addr = gnb_sock.recvfrom(4096)
    received_control["data"] = control_data
    stop_event.set()
    gnb_sock.close()
    reply_sock.close()


def test_live_kpm_source_end_to_end_over_loopback():
    ready = threading.Event()
    stop_event = threading.Event()
    received_first_request = {}
    received_control = {}

    agent = threading.Thread(
        target=_fake_gnb_agent,
        args=(ready, received_first_request, stop_event, received_control),
        daemon=True,
    )
    agent.start()
    assert ready.wait(timeout=5)

    source = LiveKpmSource(
        gnb_id="gnb-0", xapp_listen_port=XAPP_LISTEN_PORT, gnb_listen_port=GNB_LISTEN_PORT,
        recv_timeout_s=5.0,
    )
    samples = source.poll()

    assert len(samples) == 1
    assert samples[0].rnti == 7
    assert samples[0].avg_prbs_dl == pytest.approx(4.0)
    assert samples[0].gnb_id == "gnb-0"

    req_msg = ran_messages_pb2.RAN_message()
    req_msg.ParseFromString(received_first_request["data"])
    assert req_msg.msg_type == ran_messages_pb2.RAN_message_type.INDICATION_REQUEST

    source.send_control("gnb-0", sst=1, sd=1, min_ratio=5, max_ratio=25)
    assert stop_event.wait(timeout=5)
    agent.join(timeout=5)
    assert not agent.is_alive()

    ctrl_msg = ran_messages_pb2.RAN_message()
    ctrl_msg.ParseFromString(received_control["data"])
    assert ctrl_msg.msg_type == ran_messages_pb2.RAN_message_type.CONTROL
    slicing = ctrl_msg.ran_control_request.target_param_map[0].slicing_ctrl
    assert (slicing.min_ratio, slicing.max_ratio) == (5, 25)

    source.close()


def test_poll_times_out_when_no_gnb_agent_running():
    source = LiveKpmSource(
        gnb_id="gnb-0", xapp_listen_port=XAPP_LISTEN_PORT + 1, gnb_listen_port=GNB_LISTEN_PORT + 1,
        recv_timeout_s=0.3,
    )
    with pytest.raises(TimeoutError):
        source.poll()
    source.close()
