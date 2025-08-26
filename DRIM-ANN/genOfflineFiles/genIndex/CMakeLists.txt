# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(genFaissIndex EXCLUDE_FROM_ALL genFaissIndex.cpp)
target_link_libraries(genFaissIndex PRIVATE faiss)
