#!/usr/bin/env python3
"""
Generate a UE fleet docker-compose file and profile CSV.

The generated compose file is intended to be used with docker_open5gs and creates
one NR-UE container per UE with unique IMSI/IMEI identities. It supports both
UERANSIM and native OAI NR-UE stacks.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple


def _default_paths():
    root_dir = Path(__file__).resolve().parents[2]
    out_dir = root_dir / "docker_open5gs" / "generated"
    return out_dir / "nr-ue-fleet.yaml", out_dir / "ue_fleet_profiles.csv"


def _build_numeric_id(base_value: str, offset: int) -> str:
    width = len(base_value)
    value = int(base_value) + offset
    return f"{value:0{width}d}"


def _assign_profiles(
    ue_count: int, embb_ratio: float, seed: int, profile_mode: str, custom_profiles: str = "",
) -> List[str]:
    if profile_mode == "custom":
        profiles = [p.strip().lower() for p in custom_profiles.split(",") if p.strip()]
        if len(profiles) != ue_count:
            raise ValueError(
                f"--custom-profiles has {len(profiles)} entries, expected --ue-count={ue_count}"
            )
        valid = {"embb", "urllc", "mmtc"}
        bad = [p for p in profiles if p not in valid]
        if bad:
            raise ValueError(f"--custom-profiles has invalid entries {bad}, must be one of {valid}")
        return profiles

    if profile_mode == "triad":
        if ue_count % 3 != 0:
            raise ValueError("--profile-mode triad requires --ue-count to be a multiple of 3")
        # Deterministic one-per-class mapping used for phase-3 precondition runs,
        # generalised to N-per-class (ue_count=3 -> identical ["embb","urllc","mmtc"]
        # as before) so multi-UE-per-slice contention tests can reuse the same
        # deterministic, balanced assignment instead of the embb/urllc-only ratio
        # path below (which has no mmtc branch at all).
        per_slice = ue_count // 3
        return ["embb"] * per_slice + ["urllc"] * per_slice + ["mmtc"] * per_slice

    if profile_mode == "auto" and ue_count == 3:
        return ["embb", "urllc", "mmtc"]

    rng = random.Random(seed)

    if ue_count <= 1:
        return ["embb"] * ue_count

    embb_count = int(round(ue_count * embb_ratio))
    embb_count = max(1, min(ue_count - 1, embb_count))
    urllc_count = ue_count - embb_count

    profiles = ["embb"] * embb_count + ["urllc"] * urllc_count
    rng.shuffle(profiles)
    return profiles


def _profile_to_slice(profile: str) -> Tuple[str, str, str]:
    """
    Map UE profile to a deterministic S-NSSAI assignment.

    - eMBB  UEs -> SD 000000 (slice 1-0)
    - URLLC UEs -> SD 000001 (slice 1-1)
    - mMTC  UEs -> SD 000002 (slice 1-2)
    """
    sst = "1"
    profile = profile.lower()
    if profile == "embb":
        return sst, "000000", "1-0"
    if profile == "urllc":
        return sst, "000001", "1-1"
    if profile == "mmtc":
        return sst, "000002", "1-2"
    return sst, "000000", "1-0"


def _write_ueransim_compose(
    compose_path: Path,
    ue_count: int,
    profiles: List[str],
    imsi_base: str,
    imei_base: str,
    imeisv_base: str,
    ki: str,
    op: str,
    amf: str,
    gnb_ip_override: str,
) -> List[dict]:
    records = []

    lines: List[str] = []
    lines.append("version: '3'")
    lines.append("services:")

    for i in range(1, ue_count + 1):
        idx = f"{i:03d}"
        service_name = f"nr_ue_{idx}"
        component_name = f"ueransim-ue-{idx}"
        imsi = _build_numeric_id(imsi_base, i)
        imei = _build_numeric_id(imei_base, i)
        imeisv = _build_numeric_id(imeisv_base, i)
        profile = profiles[i - 1]
        sst, sd, slice_id = _profile_to_slice(profile)

        records.append(
            {
                "service_name": service_name,
                "container_name": service_name,
                "component_name": component_name,
                "imsi": imsi,
                "imei": imei,
                "imeisv": imeisv,
                "ki": ki,
                "op": op,
                "amf": amf,
                "profile": profile,
                "sst": sst,
                "sd": sd,
                "slice_id": slice_id,
            }
        )

        env_lines = [
            f"      - COMPONENT_NAME={component_name}",
            f"      - UE1_IMSI={imsi}",
            f"      - UE1_IMEI={imei}",
            f"      - UE1_IMEISV={imeisv}",
            f"      - UE1_KI={ki}",
            f"      - UE1_OP={op}",
            f"      - UE1_AMF={amf}",
            f"      - UE_PROFILE={profile}",
            f"      - UE_SLICE_SST={sst}",
            f"      - UE_SLICE_SD={sd}",
        ]
        if gnb_ip_override:
            env_lines.append(f"      - NR_GNB_IP={gnb_ip_override}")

        lines.extend(
            [
                f"  {service_name}:",
                "    image: docker_ueransim",
                f"    container_name: {service_name}",
                "    stdin_open: true",
                "    tty: true",
                "    volumes:",
                "      - ../ueransim:/mnt/ueransim",
                "      - /etc/timezone:/etc/timezone:ro",
                "      - /etc/localtime:/etc/localtime:ro",
                "    env_file:",
                "      - ../.env",
                "    environment:",
                *env_lines,
                "    expose:",
                '      - "4997/udp"',
                "    cap_add:",
                "      - NET_ADMIN",
                "    privileged: true",
                "    networks:",
                "      default: {}",
            ]
        )

    lines.extend(
        [
            "networks:",
            "  default:",
            "    external:",
            "      name: docker_open5gs_default",
        ]
    )

    compose_path.parent.mkdir(parents=True, exist_ok=True)
    compose_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def _write_oai_compose(
    compose_path: Path,
    ue_count: int,
    profiles: List[str],
    imsi_base: str,
    imei_base: str,
    imeisv_base: str,
    ki: str,
    op: str,
    amf: str,
    gnb_ip_override: str,
    oai_ue_image: str,
    oai_launch_mode: str,
) -> List[dict]:
    records = []

    lines: List[str] = []
    lines.append("version: '3'")
    lines.append("services:")

    for i in range(1, ue_count + 1):
        idx = f"{i:03d}"
        service_name = f"nr_ue_{idx}"
        component_name = f"oai-ue-{idx}"
        imsi = _build_numeric_id(imsi_base, i)
        imei = _build_numeric_id(imei_base, i)
        imeisv = _build_numeric_id(imeisv_base, i)
        profile = profiles[i - 1]
        sst, sd, slice_id = _profile_to_slice(profile)

        records.append(
            {
                "service_name": service_name,
                "container_name": service_name,
                "component_name": component_name,
                "imsi": imsi,
                "imei": imei,
                "imeisv": imeisv,
                "ki": ki,
                "op": op,
                "amf": amf,
                "profile": profile,
                "sst": sst,
                "sd": sd,
                "slice_id": slice_id,
            }
        )

        extra_options = (
            f"--sa --rfsim -r 106 --numerology 1 --band 78 -C 3619200000 "
            f"--uicc0.imsi {imsi} --uicc0.nssai_sd {sd} --uicc0.dnn internet --uicc0.nssai_sst {sst} "
            f"--rfsimulator.serveraddr {gnb_ip_override} --log_config.global_log_options level,nocolor,time"
        )

        if oai_launch_mode == "local-build":
            launch_cmd = (
                "cd /openairinterface5g && . ./oaienv && cd cmake_targets/ran_build/build && "
                f"./nr-uesoftmodem -O /tmp/nr-ue.conf {extra_options}"
            )
            lines.extend(
                [
                    f"  {service_name}:",
                    f"    image: {oai_ue_image}",
                    f"    container_name: {service_name}",
                    "    entrypoint: /bin/bash",
                    "    command:",
                    "      - -lc",
                    f"      - \"{launch_cmd}\"",
                    "    stdin_open: true",
                    "    tty: true",
                    "    cap_add:",
                    "      - NET_ADMIN",
                    "      - NET_RAW",
                    "      - SYS_NICE",
                    "      - IPC_LOCK",
                    "    devices:",
                    "      - /dev/net/tun:/dev/net/tun",
                    "    volumes:",
                    "      - ../oai/nrue.uicc.conf:/tmp/nr-ue.conf:ro",
                    "      - /etc/timezone:/etc/timezone:ro",
                    "      - /etc/localtime:/etc/localtime:ro",
                    "    networks:",
                    "      default: {}",
                ]
            )
        else:
            env_options = f"-E {extra_options}"
            lines.extend(
                [
                    f"  {service_name}:",
                    f"    image: {oai_ue_image}",
                    f"    container_name: {service_name}",
                    "    stdin_open: true",
                    "    tty: true",
                    "    cap_add:",
                    "      - NET_ADMIN",
                    "      - NET_RAW",
                    "      - SYS_NICE",
                    "      - IPC_LOCK",
                    "    devices:",
                    "      - /dev/net/tun:/dev/net/tun",
                    "    volumes:",
                    "      - ../oai/nrue.uicc.conf:/opt/oai-nr-ue/etc/nr-ue.conf",
                    "      - /etc/timezone:/etc/timezone:ro",
                    "      - /etc/localtime:/etc/localtime:ro",
                    "    environment:",
                    f"      - USE_ADDITIONAL_OPTIONS={env_options}",
                    "    networks:",
                    "      default: {}",
                ]
            )

    lines.extend(
        [
            "networks:",
            "  default:",
            "    external:",
            "      name: docker_open5gs_default",
        ]
    )

    compose_path.parent.mkdir(parents=True, exist_ok=True)
    compose_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def _write_profiles_csv(csv_path: Path, records: List[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "service_name",
                "container_name",
                "component_name",
                "imsi",
                "imei",
                "imeisv",
                "ki",
                "op",
                "amf",
                "profile",
                "sst",
                "sd",
                "slice_id",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def _write_compose(
    compose_path: Path,
    ue_count: int,
    profiles: List[str],
    imsi_base: str,
    imei_base: str,
    imeisv_base: str,
    ki: str,
    op: str,
    amf: str,
    gnb_ip_override: str,
    ue_stack: str,
    oai_ue_image: str,
    oai_launch_mode: str,
) -> List[dict]:
    if ue_stack == "ueransim":
        return _write_ueransim_compose(
            compose_path=compose_path,
            ue_count=ue_count,
            profiles=profiles,
            imsi_base=imsi_base,
            imei_base=imei_base,
            imeisv_base=imeisv_base,
            ki=ki,
            op=op,
            amf=amf,
            gnb_ip_override=gnb_ip_override,
        )
    if ue_stack == "oai":
        return _write_oai_compose(
            compose_path=compose_path,
            ue_count=ue_count,
            profiles=profiles,
            imsi_base=imsi_base,
            imei_base=imei_base,
            imeisv_base=imeisv_base,
            ki=ki,
            op=op,
            amf=amf,
            gnb_ip_override=gnb_ip_override,
            oai_ue_image=oai_ue_image,
            oai_launch_mode=oai_launch_mode,
        )

    raise ValueError(f"Unsupported UE stack: {ue_stack}")


def parse_args() -> argparse.Namespace:
    default_compose, default_profiles = _default_paths()

    parser = argparse.ArgumentParser(description="Generate 5G UE fleet compose + profile mapping")
    parser.add_argument("--ue-count", type=int, default=100, help="Number of UE containers")
    parser.add_argument("--embb-ratio", type=float, default=0.7, help="Fraction of eMBB UEs")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for profile assignment")
    parser.add_argument(
        "--profile-mode",
        choices=["auto", "ratio", "triad", "custom"],
        default="auto",
        help="UE profile assignment mode (auto uses triad for 3 UEs, otherwise embb/urllc ratio; "
             "custom uses --custom-profiles verbatim, for uneven per-slice contention tests)",
    )
    parser.add_argument(
        "--custom-profiles",
        default="",
        help="Comma-separated profile list (e.g. embb,urllc,urllc,mmtc), used only with "
             "--profile-mode custom. Length must equal --ue-count.",
    )

    parser.add_argument("--imsi-base", default="001010000100000", help="Base IMSI as integer string")
    parser.add_argument("--imei-base", default="356938035640000", help="Base IMEI as integer string")
    parser.add_argument("--imeisv-base", default="4370816125800000", help="Base IMEISV as integer string")

    parser.add_argument("--ki", default="465B5CE8B199B49FAA5F0A2EE238A6BC", help="UE Ki")
    parser.add_argument("--op", default="E8ED289DEBA952E4283B54E88E6183CA", help="UE OP/OPC")
    parser.add_argument("--amf", default="8000", help="UE AMF")
    parser.add_argument(
        "--gnb-ip-override",
        default="",
        help="Optional gNB IP to inject into each UE container (overrides env_file NR_GNB_IP)",
    )

    parser.add_argument("--output-compose", default=str(default_compose), help="Output docker-compose path")
    parser.add_argument("--ue-stack", choices=["ueransim", "oai"], default="ueransim", help="Type of UE containers to generate")
    parser.add_argument(
        "--oai-ue-image",
        default="oaisoftwarealliance/oai-nr-ue:develop",
        help="OAI UE container image (used when --ue-stack oai)",
    )
    parser.add_argument(
        "--oai-launch-mode",
        choices=["env", "local-build"],
        default="env",
        help="OAI UE launch mode: env (USE_ADDITIONAL_OPTIONS) or local-build (nr-uesoftmodem from source tree)",
    )
    parser.add_argument("--output-profiles", default=str(default_profiles), help="Output profiles CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.ue_count <= 0:
        raise ValueError("--ue-count must be > 0")
    if not (0.0 <= args.embb_ratio <= 1.0):
        raise ValueError("--embb-ratio must be between 0 and 1")

    compose_path = Path(args.output_compose).resolve()
    csv_path = Path(args.output_profiles).resolve()

    profiles = _assign_profiles(
        args.ue_count, args.embb_ratio, args.random_seed, args.profile_mode, args.custom_profiles,
    )
    records = _write_compose(
        compose_path=compose_path,
        ue_count=args.ue_count,
        profiles=profiles,
        imsi_base=args.imsi_base,
        imei_base=args.imei_base,
        imeisv_base=args.imeisv_base,
        ki=args.ki,
        op=args.op,
        amf=args.amf,
        gnb_ip_override=args.gnb_ip_override,
        ue_stack=args.ue_stack,
        oai_ue_image=args.oai_ue_image,
        oai_launch_mode=args.oai_launch_mode,
    )
    _write_profiles_csv(csv_path, records)

    embb_count = sum(1 for record in records if record["profile"] == "embb")
    urllc_count = sum(1 for record in records if record["profile"] == "urllc")
    mmtc_count = sum(1 for record in records if record["profile"] == "mmtc")

    print(f"Generated compose file: {compose_path}")
    print(f"Generated profile CSV : {csv_path}")
    print(f"UE count={len(records)} | eMBB={embb_count} | URLLC={urllc_count} | mMTC={mmtc_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
