#include "xla/symbolic_expr.h"

#include "xla/xla_data.pb.h"
#include "gtest/gtest.h"

namespace xla::symbolic_expr {
namespace {

TEST(SymbolicExprTest, StructuralAndNormalizedEqualityDiffer) {
  ExprHandle lhs = Add(Variable(1), Constant(3));
  ExprHandle rhs = Add(Constant(3), Variable(1));

  EXPECT_FALSE(StructuralEqual(lhs, rhs));
  EXPECT_TRUE(Equal(lhs, rhs));
}

TEST(SymbolicExprTest, SimplifyFoldsCoefficients) {
  ExprHandle expr = Mul(Constant(3), Mul(Constant(4), Variable(1)));
  ExprHandle simplified = Simplify(expr);

  EXPECT_TRUE(Equal(simplified, Mul(Constant(12), Variable(1))));
  EXPECT_EQ(ToString(simplified), "(12 * Var(1))");
}

TEST(SymbolicExprTest, SubstituteAndEvaluate) {
  ExprHandle expr = Mul(Add(Variable(1), Constant(2)), Constant(3));
  ExprHandle substituted = Substitute(expr, 1, Constant(4));

  EXPECT_EQ(Evaluate(Simplify(substituted)), 18);
}

TEST(SymbolicExprTest, EvaluateRejectsDivisionByZero) {
  ExprHandle expr = Div(Constant(8), Constant(0));

  EXPECT_FALSE(Evaluate(Simplify(expr)).has_value());
}

TEST(SymbolicExprTest, ProtoRoundTripWithXlaProto) {
  ExprHandle expr = Add(Mul(Constant(2), Variable(7)), Constant(5));
  xla::ExpressionProto proto;

  ToProto(expr, &proto);
  ExprHandle decoded = FromProto(proto);

  EXPECT_TRUE(Equal(expr, decoded));
  EXPECT_EQ(ToString(decoded), "((2 * Var(7)) + 5)");
}

}  // namespace
}  // namespace xla::symbolic_expr