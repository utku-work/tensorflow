#ifndef XLA_SYMBOLIC_EXPR_H_
#define XLA_SYMBOLIC_EXPR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace xla::symbolic_expr {

class Expr;
using ExprHandle = std::shared_ptr<const Expr>;

class Expr final {
 public:
  enum class Kind : uint8_t {
    kConstant,
    kVariable,
    kAdd,
    kSub,
    kMul,
    kDiv,
  };

  Kind kind() const { return kind_; }
  bool is_constant() const { return kind_ == Kind::kConstant; }
  bool is_variable() const { return kind_ == Kind::kVariable; }
  bool is_binary() const {
    return kind_ == Kind::kAdd || kind_ == Kind::kSub || kind_ == Kind::kMul ||
           kind_ == Kind::kDiv;
  }

  int64_t constant_value() const { return value_; }
  int32_t variable_id() const { return static_cast<int32_t>(value_); }
  const ExprHandle& lhs() const { return lhs_; }
  const ExprHandle& rhs() const { return rhs_; }

 private:
  friend ExprHandle Constant(int64_t value);
  friend ExprHandle Variable(int32_t id);
  friend ExprHandle Add(ExprHandle lhs, ExprHandle rhs);
  friend ExprHandle Sub(ExprHandle lhs, ExprHandle rhs);
  friend ExprHandle Mul(ExprHandle lhs, ExprHandle rhs);
  friend ExprHandle Div(ExprHandle lhs, ExprHandle rhs);

  Expr(Kind kind, int64_t value, ExprHandle lhs, ExprHandle rhs)
      : kind_(kind), value_(value), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

  Kind kind_;
  int64_t value_;
  ExprHandle lhs_;
  ExprHandle rhs_;
};

ExprHandle Constant(int64_t value);
ExprHandle Variable(int32_t id);
ExprHandle Add(ExprHandle lhs, ExprHandle rhs);
ExprHandle Sub(ExprHandle lhs, ExprHandle rhs);
ExprHandle Mul(ExprHandle lhs, ExprHandle rhs);
ExprHandle Div(ExprHandle lhs, ExprHandle rhs);

bool StructuralEqual(const ExprHandle& lhs, const ExprHandle& rhs);
bool Equal(const ExprHandle& lhs, const ExprHandle& rhs);

ExprHandle Simplify(const ExprHandle& expr);
ExprHandle Substitute(const ExprHandle& expr, int32_t variable_id,
                      const ExprHandle& replacement);
std::optional<int64_t> Evaluate(const ExprHandle& expr);
std::string ToString(const ExprHandle& expr);

template <typename Proto>
void ToProto(const ExprHandle& expr, Proto* proto) {
  proto->Clear();
  if (!expr) {
    return;
  }

  switch (expr->kind()) {
    case Expr::Kind::kConstant:
      proto->set_constant_value(expr->constant_value());
      return;
    case Expr::Kind::kVariable:
      proto->set_variable_id(expr->variable_id());
      return;
    case Expr::Kind::kAdd: {
      auto* add_node = proto->mutable_add_node();
      ToProto(expr->lhs(), add_node->mutable_lhs());
      ToProto(expr->rhs(), add_node->mutable_rhs());
      return;
    }
    case Expr::Kind::kSub: {
      auto* sub_node = proto->mutable_sub_node();
      ToProto(expr->lhs(), sub_node->mutable_lhs());
      ToProto(expr->rhs(), sub_node->mutable_rhs());
      return;
    }
    case Expr::Kind::kMul: {
      auto* mul_node = proto->mutable_mul_node();
      ToProto(expr->lhs(), mul_node->mutable_lhs());
      ToProto(expr->rhs(), mul_node->mutable_rhs());
      return;
    }
    case Expr::Kind::kDiv: {
      auto* div_node = proto->mutable_div_node();
      ToProto(expr->lhs(), div_node->mutable_lhs());
      ToProto(expr->rhs(), div_node->mutable_rhs());
      return;
    }
  }
}

template <typename Proto>
ExprHandle FromProto(const Proto& proto) {
  switch (proto.node_type_case()) {
    case Proto::kConstantValue:
      return Constant(proto.constant_value());
    case Proto::kVariableId:
      return Variable(proto.variable_id());
    case Proto::kAddNode:
      return Add(FromProto(proto.add_node().lhs()),
                 FromProto(proto.add_node().rhs()));
    case Proto::kSubNode:
      return Sub(FromProto(proto.sub_node().lhs()),
                 FromProto(proto.sub_node().rhs()));
    case Proto::kMulNode:
      return Mul(FromProto(proto.mul_node().lhs()),
                 FromProto(proto.mul_node().rhs()));
    case Proto::kDivNode:
      return Div(FromProto(proto.div_node().lhs()),
                 FromProto(proto.div_node().rhs()));
    case Proto::NODE_TYPE_NOT_SET:
    default:
      return nullptr;
  }
}

}  // namespace xla::symbolic_expr

#endif  // XLA_SYMBOLIC_EXPR_H_