/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_cost_analysis.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array4d.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/padding.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/service.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// This test suite tests the HLO cost analysis by first building a computation
// using the client computation builder and running the HloCostAnalysis that
// returns the number of floating point and transcendental operations in the
// graph. We test both individual HLO operations as well as a mixed graph.
class HloCostAnalysisTest : public ::testing::Test {
 protected:
  HloCostAnalysisTest()
      : client_(ClientLibrary::LocalClientOrDie()),
        // Accessing service instance is required for the unit tests to enable
        // whitebox accesses to the user computation built from the client,
        // as shown in the BuildHloGraph functions below.
        service_(static_cast<Service*>(ClientLibrary::GetXlaService(
            static_cast<LocalClient*>(client_)->platform()))) {
    // Create a computation for a unary user function: x => exp(x + 0.5)
    {
      XlaBuilder builder("add_and_exp");
      auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
      auto half = ConstantR0<float>(&builder, 0.5);
      Exp(Add(x, half));
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      add_and_exp_ = std::move(computation_status).value();
    }

    // Create a computation for a binary user function: (x, y) => x + y
    {
      XlaBuilder builder("add");
      auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
      auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
      Add(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      add_ = std::move(computation_status).value();
    }

    // Create a computation for a sigmoid function: x => 1 / (1 + exp(-x))
    {
      XlaBuilder builder("sigmoid");
      auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
      auto one = ConstantR0<float>(&builder, 1.0);
      Div(one, Add(one, Exp(Neg(x))));
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      sigmoid_ = std::move(computation_status).value();
    }

    // Create a computation for a binary max function: (x, y) => max (x, y)
    {
      XlaBuilder builder("max");
      auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
      auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
      Max(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      max_ = std::move(computation_status).value();
    }

    // Create a computation for a binary GT function: (x, y) => x > y
    {
      XlaBuilder builder("gt");
      auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
      auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
      Gt(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      gt_ = std::move(computation_status).value();
    }
  }

  // Build HLO graph from the given builder and return the HLO module.
  std::unique_ptr<HloModule> BuildHloGraph(XlaBuilder* builder) {
    auto computation_status = builder->Build();
    TF_CHECK_OK(computation_status.status());
    auto computation = std::move(computation_status).value();
    auto config = HloModule::CreateModuleConfigFromProto(computation.proto(),
                                                         DebugOptions())
                      .value();
    return HloModule::CreateFromProto(computation.proto(), config).value();
  }

  Client* client_;
  Service* service_;

  // User computations used for higher order operations (e.g., Map, Reduce).
  XlaComputation add_;
  XlaComputation add_and_exp_;
  XlaComputation sigmoid_;
  XlaComputation max_;
  XlaComputation gt_;
};


}  // namespace
}  // namespace xla
