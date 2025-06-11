#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import codecs
import os
import re
import sys

import cpplint
import pycodestyle
from cpplint import _cpplint_state

CXX_SUFFIX = set(["cc", "c", "cpp", "h", "cu", "hpp"])


def filepath_enumerate(paths):
    """Enumerate the file paths of all subfiles of the list of paths."""
    out = []
    for path in paths:
        if os.path.isfile(path):
            out.append(path)
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    out.append(os.path.normpath(os.path.join(root, name)))
    return out


class LintHelper(object):
    @staticmethod
    def _print_summary_map(strm, result_map, ftype):
        """Print summary of certain result map."""
        if len(result_map) == 0:
            return 0
        npass = len([x for k, x in result_map.items() if len(x) == 0])
        strm.write(
            "=====%d/%d %s files passed check=====\n"
            % (npass, len(result_map), ftype)
        )
        for fname, emap in result_map.items():
            if len(emap) == 0:
                continue
            strm.write(
                "%s: %d Errors of %d Categories map=%s\n"
                % (fname, sum(emap.values()), len(emap), str(emap))
            )
        return len(result_map) - npass

    def __init__(self) -> None:
        self.project_name = None
        self.cpp_header_map = {}
        self.cpp_src_map = {}
        super().__init__()
        cpplint_args = [".", "--extensions=" + (",".join(CXX_SUFFIX))]
        _ = cpplint.ParseArguments(cpplint_args)
        cpplint._SetFilters(
            ",".join(
                [
                    "-build/c++11",
                    "-build/namespaces",
                    "-build/include,",
                    "+build/include_what_you_use",
                    "+build/include_order",
                ]
            )
        )
        cpplint._SetCountingStyle("toplevel")
        cpplint._line_length = 80

    def process_cpp(self, path, suffix):
        """Process a cpp file."""
        _cpplint_state.ResetErrorCounts()
        cpplint.ProcessFile(str(path), _cpplint_state.verbose_level)
        _cpplint_state.PrintErrorCounts()
        errors = _cpplint_state.errors_by_category.copy()

        if suffix == "h":
            self.cpp_header_map[str(path)] = errors
        else:
            self.cpp_src_map[str(path)] = errors

    def print_summary(self, strm):
        """Print summary of lint."""
        nerr = 0
        nerr += LintHelper._print_summary_map(
            strm, self.cpp_header_map, "cpp-header"
        )
        nerr += LintHelper._print_summary_map(
            strm, self.cpp_src_map, "cpp-source"
        )
        if nerr == 0:
            strm.write("All passed!\n")
        else:
            strm.write("%d files failed lint\n" % nerr)
        return nerr


# singleton helper for lint check
_HELPER = LintHelper()


def process(fname, allow_type):
    """Process a file."""
    fname = str(fname)
    arr = fname.rsplit(".", 1)
    if fname.find("#") != -1 or arr[-1] not in allow_type:
        return
    if arr[-1] in CXX_SUFFIX:
        _HELPER.process_cpp(fname, arr[-1])


def main():
    """Main entry function."""
    parser = argparse.ArgumentParser(description="lint source codes")
    parser.add_argument("--project", help="project name")
    parser.add_argument(
        "--path",
        nargs="+",
        default=[],
        help="path to traverse",
        required=False,
    )
    parser.add_argument(
        "--exclude_path",
        nargs="+",
        default=[],
        help="exclude this path, and all subfolders " + "if path is a folder",
    )

    args = parser.parse_args()
    _HELPER.project_name = args.project
    allow_type = []
    allow_type += [x for x in CXX_SUFFIX]
    allow_type = set(allow_type)

    # get excluded files
    excluded_paths = filepath_enumerate(args.exclude_path)
    for path in args.path:
        if os.path.isfile(path):
            normpath = os.path.normpath(path)
            if normpath not in excluded_paths:
                process(path, allow_type)
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    file_path = os.path.normpath(os.path.join(root, name))
                    if file_path not in excluded_paths:
                        process(file_path, allow_type)
    nerr = _HELPER.print_summary(sys.stderr)
    sys.exit(nerr > 0)


if __name__ == "__main__":
    main()
