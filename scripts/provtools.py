# -*- coding: utf-8 -*-
"""
This package provides an interface to command-line tools for PROV documents.

Author: Trung Dong Huynh
"""
import json
import logging
from pathlib import Path
import subprocess

import environs

# Reading local environment variables
env = environs.Env()
env.read_env()

PROVCONVERT_PATH = env("PROVCONVERT", default="provconvert")
PROVMAN_PATH = env("PROVMAN", default="provmanagement")
logger = logging.getLogger(__name__)


def call_external_tool(executable, arguments, pipe_input=None, timeout=None) -> str:
    """Call the external command-line tool at `executable` with the provide arguments.

    Args:
        executable:
        arguments:
        pipe_input:
        timeout:

    Returns: The output from the execution of the tool.
    """
    args = list(map(str, (executable, *arguments)))
    logger.debug("Calling command: %s", " ".join(args))
    p = subprocess.Popen(
        args,
        stdin=subprocess.PIPE if pipe_input is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    b_input = pipe_input.encode("utf8") if pipe_input is not None else None
    stdout, stderr = p.communicate(input=b_input, timeout=timeout)
    if p.returncode != 0:
        stdout, stderr = stdout.decode(), stderr.decode()
        logger.debug(
            "%s returns non-zero code (%d)\nOutput: %s\nError: %s",
            executable,
            p.returncode,
            stdout,
            stderr,
        )
        raise subprocess.CalledProcessError(p.returncode, args, stdout, stderr)
    return stdout.decode()


def provconvert_file(from_filepath, to_filepath):
    """Convert a PROV document into another using provconvert.

    The format of the input and output files will be guessed
    from their file extension.
    """
    call_external_tool(
        PROVCONVERT_PATH, ["-infile", from_filepath, "-outfile", to_filepath]
    )


def provconvert_merge(filepaths, to_filepath: str):
    call_external_tool(
        PROVCONVERT_PATH,
        ["-merge", "-", "-outfile", to_filepath, "-flatten"],
        "\n".join(filepaths),
    )


def provman_kernelize(
    filepath: Path,
    outpath: Path,
    file_id: str,
    level_from: int = 0,
    level_to: int = 2,
    level0: str = None,
    no_primitives: bool = False,
    readable_types: bool = False,
    aggregating_types: bool = False,
    triangle: bool = False,
    return_cmd_only: bool = False,
):
    out_filepath = outpath / (file_id + "-%level.features.json")
    cmd_line_args = [
        "kernelize",
        "--infile",
        filepath,
        "--from",
        level_from,
        "--to",
        level_to,
        "-F",
        out_filepath,
    ]

    if level0 is not None:
        cmd_line_args.extend(["--level0", level0])

    if no_primitives:
        cmd_line_args.append("-n")

    if readable_types:
        cmd_line_args.append("-R")

    if not aggregating_types:
        cmd_line_args.extend(["--aggregate", "false"])

    if triangle:
        cmd_line_args.extend(["--triangle", "true"])

    if return_cmd_only:
        return " ".join(map(str, cmd_line_args))

    call_external_tool(PROVMAN_PATH, cmd_line_args)


def provman_batch(commands: str):
    call_external_tool(PROVMAN_PATH, ["batch", "--infile", "-", "--outfile", "ignore", "--parallel", "true"], pipe_input=commands)
