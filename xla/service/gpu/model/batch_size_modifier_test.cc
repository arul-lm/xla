/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string>
#include "xla/service/gpu/model/batch_size_modifier_core.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

TEST(BatchSizeModifierTest, RewritesExplicitBatchDimension) {
  std::string tmp = tsl::testing::TmpDir();
  std::string yaml = tsl::io::JoinPath(tmp, "cfg.yaml");
  // Path must contain SPxEPxTP for mesh inference (here 1x2x1).
  std::string in_path = tsl::io::JoinPath(tmp, "case_1x2x1.mlir");
  std::string out_path = tsl::io::JoinPath(tmp, "out.mlir");

  TF_EXPECT_OK(tsl::WriteStringToFile(tsl::Env::Default(), yaml,
                                     R"(modify_batch_size:
  num_experts: 8
  seq_len: 4
  num_experts_per_tok: 2
  sp: 1
)"));
  TF_EXPECT_OK(tsl::WriteStringToFile(
      tsl::Env::Default(), in_path,
      "%x = f32[2,4] parameter(0)\n"));

  BatchSizeModifierOptions opts;
  opts.input_mlir_path = in_path;
  opts.output_mlir_path = out_path;
  opts.old_batch_size = 2;
  opts.new_batch_size = 8;
  opts.config_yaml_path = yaml;
  opts.path_for_mesh_inference = in_path;
  opts.enable_reshape_fix = true;
  opts.strict_mode = false;

  TF_EXPECT_OK(RunBatchSizeModification(opts));

  std::string out;
  TF_EXPECT_OK(tsl::ReadFileToString(tsl::Env::Default(), out_path, &out));
  EXPECT_NE(out.find("f32[8,4]"), std::string::npos);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
