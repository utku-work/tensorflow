#include "xla/symbolic_expr.h"

#include <utility>

namespace xla::symbolic_expr {

namespace {

bool IsConstantValue(const ExprHandle& expr, int64_t value) {
  return expr && expr->is_constant() && expr->constant_value() == value;
}

bool EquivalentImpl(const ExprHandle& lhs, const ExprHandle& rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->kind() != rhs->kind()) {
    return false;
  }

  switch (lhs->kind()) {
    case Expr::Kind::kConstant:
      return lhs->constant_value() == rhs->constant_value();
    case Expr::Kind::kVariable:
      return lhs->variable_id() == rhs->variable_id();
    case Expr::Kind::kAdd:
    case Expr::Kind::kMul:
      return (EquivalentImpl(lhs->lhs(), rhs->lhs()) &&
              EquivalentImpl(lhs->rhs(), rhs->rhs())) ||
             (EquivalentImpl(lhs->lhs(), rhs->rhs()) &&
              EquivalentImpl(lhs->rhs(), rhs->lhs()));
    case Expr::Kind::kSub:
    case Expr::Kind::kDiv:
      return EquivalentImpl(lhs->lhs(), rhs->lhs()) &&
             EquivalentImpl(lhs->rhs(), rhs->rhs());
  }
}

std::string ToStringImpl(const ExprHandle& expr) {
  if (!expr) {
    return "<null>";
  }

  switch (expr->kind()) {
    case Expr::Kind::kConstant:
      return std::to_string(expr->constant_value());
    case Expr::Kind::kVariable:
      return "Var(" + std::to_string(expr->variable_id()) + ")";
    case Expr::Kind::kAdd:
      return "(" + ToStringImpl(expr->lhs()) + " + " +
             ToStringImpl(expr->rhs()) + ")";
    case Expr::Kind::kSub:
      return "(" + ToStringImpl(expr->lhs()) + " - " +
             ToStringImpl(expr->rhs()) + ")";
    case Expr::Kind::kMul:
      return "(" + ToStringImpl(expr->lhs()) + " * " +
             ToStringImpl(expr->rhs()) + ")";
    case Expr::Kind::kDiv:
      return "(" + ToStringImpl(expr->lhs()) + " / " +
             ToStringImpl(expr->rhs()) + ")";
  }
}

}  // namespace

ExprHandle Constant(int64_t value) {
  return std::make_shared<Expr>(Expr::Kind::kConstant, value, nullptr,
                                nullptr);
}

ExprHandle Variable(int32_t id) {
  return std::make_shared<Expr>(Expr::Kind::kVariable, id, nullptr, nullptr);
}

ExprHandle Add(ExprHandle lhs, ExprHandle rhs) {
  return std::make_shared<Expr>(Expr::Kind::kAdd, 0, std::move(lhs),
                                std::move(rhs));
}

ExprHandle Sub(ExprHandle lhs, ExprHandle rhs) {
  return std::make_shared<Expr>(Expr::Kind::kSub, 0, std::move(lhs),
                                std::move(rhs));
}

ExprHandle Mul(ExprHandle lhs, ExprHandle rhs) {
  return std::make_shared<Expr>(Expr::Kind::kMul, 0, std::move(lhs),
                                std::move(rhs));
}

ExprHandle Div(ExprHandle lhs, ExprHandle rhs) {
  return std::make_shared<Expr>(Expr::Kind::kDiv, 0, std::move(lhs),
                                std::move(rhs));
}

bool StructuralEqual(const ExprHandle& lhs, const ExprHandle& rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->kind() != rhs->kind()) {
    return false;
  }

  switch (lhs->kind()) {
    case Expr::Kind::kConstant:
      return lhs->constant_value() == rhs->constant_value();
    case Expr::Kind::kVariable:
      return lhs->variable_id() == rhs->variable_id();
    case Expr::Kind::kAdd:
    case Expr::Kind::kSub:
    case Expr::Kind::kMul:
    case Expr::Kind::kDiv:
      return StructuralEqual(lhs->lhs(), rhs->lhs()) &&
             StructuralEqual(lhs->rhs(), rhs->rhs());
  }
}

bool Equal(const ExprHandle& lhs, const ExprHandle& rhs) {
  return EquivalentImpl(Simplify(lhs), Simplify(rhs));
}

ExprHandle Simplify(const ExprHandle& expr) {
  if (!expr || expr->is_constant() || expr->is_variable()) {
    return expr;
  }

  ExprHandle lhs = Simplify(expr->lhs());
  ExprHandle rhs = Simplify(expr->rhs());

  switch (expr->kind()) {
    case Expr::Kind::kAdd: {
      if (lhs->is_constant() && rhs->is_constant()) {
        return Constant(lhs->constant_value() + rhs->constant_value());
      }
      if (IsConstantValue(lhs, 0)) {
        return rhs;
      }
      if (IsConstantValue(rhs, 0)) {
        return lhs;
      }
      if (lhs->is_constant() && !rhs->is_constant()) {
        return Simplify(Add(rhs, lhs));
      }
      if (EquivalentImpl(lhs, rhs)) {
        return Simplify(Mul(Constant(2), rhs));
      }
      if (lhs->kind() == Expr::Kind::kMul && lhs->lhs()->is_constant() &&
          EquivalentImpl(lhs->rhs(), rhs)) {
        return Simplify(
            Mul(Simplify(Add(lhs->lhs(), Constant(1))), lhs->rhs()));
      }
      if (rhs->kind() == Expr::Kind::kMul && rhs->lhs()->is_constant() &&
          EquivalentImpl(rhs->rhs(), lhs)) {
        return Simplify(
            Mul(Simplify(Add(rhs->lhs(), Constant(1))), rhs->rhs()));
      }
      if (lhs->kind() == Expr::Kind::kMul && rhs->kind() == Expr::Kind::kMul &&
          lhs->lhs()->is_constant() && rhs->lhs()->is_constant() &&
          EquivalentImpl(lhs->rhs(), rhs->rhs())) {
        return Simplify(Mul(Simplify(Add(lhs->lhs(), rhs->lhs())), lhs->rhs()));
      }
      if (lhs->kind() == Expr::Kind::kAdd) {
        return Simplify(Add(lhs->lhs(), Add(lhs->rhs(), rhs)));
      }
      if (lhs->kind() == Expr::Kind::kSub) {
        return Simplify(Sub(lhs->lhs(), Sub(lhs->rhs(), rhs)));
      }
      return Add(lhs, rhs);
    }
    case Expr::Kind::kSub: {
      if (lhs->is_constant() && rhs->is_constant()) {
        return Constant(lhs->constant_value() - rhs->constant_value());
      }
      if (IsConstantValue(rhs, 0)) {
        return lhs;
      }
      if (EquivalentImpl(lhs, rhs)) {
        return Constant(0);
      }
      if (lhs->kind() == Expr::Kind::kMul && rhs->kind() == Expr::Kind::kMul &&
          lhs->lhs()->is_constant() && rhs->lhs()->is_constant() &&
          EquivalentImpl(lhs->rhs(), rhs->rhs())) {
        return Simplify(Mul(Simplify(Sub(lhs->lhs(), rhs->lhs())), lhs->rhs()));
      }
      if (lhs->kind() == Expr::Kind::kAdd) {
        return Simplify(Add(lhs->lhs(), Sub(lhs->rhs(), rhs)));
      }
      if (lhs->kind() == Expr::Kind::kSub) {
        return Simplify(Sub(lhs->lhs(), Add(lhs->rhs(), rhs)));
      }
      return Sub(lhs, rhs);
    }
    case Expr::Kind::kMul: {
      if (lhs->is_constant() && rhs->is_constant()) {
        return Constant(lhs->constant_value() * rhs->constant_value());
      }
      if (IsConstantValue(lhs, 0) || IsConstantValue(rhs, 0)) {
        return Constant(0);
      }
      if (IsConstantValue(lhs, 1)) {
        return rhs;
      }
      if (IsConstantValue(rhs, 1)) {
        return lhs;
      }
      if (!lhs->is_constant() && rhs->is_constant()) {
        return Simplify(Mul(rhs, lhs));
      }
      if (lhs->is_constant() && rhs->kind() == Expr::Kind::kMul &&
          rhs->lhs()->is_constant()) {
        return Simplify(
            Mul(Constant(lhs->constant_value() * rhs->lhs()->constant_value()),
                rhs->rhs()));
      }
      return Mul(lhs, rhs);
    }
    case Expr::Kind::kDiv: {
      if (lhs->is_constant() && rhs->is_constant()) {
        if (rhs->constant_value() == 0) {
          return Div(lhs, rhs);
        }
        return Constant(lhs->constant_value() / rhs->constant_value());
      }
      if (IsConstantValue(rhs, 1)) {
        return lhs;
      }
      if (lhs->kind() == Expr::Kind::kAdd) {
        return Simplify(Add(Div(lhs->lhs(), rhs), Div(lhs->rhs(), rhs)));
      }
      if (lhs->kind() == Expr::Kind::kMul) {
        return Simplify(Mul(Div(lhs->lhs(), rhs), lhs->rhs()));
      }
      if (lhs->kind() == Expr::Kind::kDiv) {
        return Simplify(Div(lhs->lhs(), Mul(lhs->rhs(), rhs)));
      }
      return Div(lhs, rhs);
    }
    case Expr::Kind::kConstant:
    case Expr::Kind::kVariable:
      return expr;
  }
}

ExprHandle Substitute(const ExprHandle& expr, int32_t variable_id,
                      const ExprHandle& replacement) {
  if (!expr) {
    return nullptr;
  }
  if (expr->is_constant()) {
    return expr;
  }
  if (expr->is_variable()) {
    return expr->variable_id() == variable_id ? replacement : expr;
  }

  ExprHandle lhs = Substitute(expr->lhs(), variable_id, replacement);
  ExprHandle rhs = Substitute(expr->rhs(), variable_id, replacement);
  switch (expr->kind()) {
    case Expr::Kind::kAdd:
      return Add(lhs, rhs);
    case Expr::Kind::kSub:
      return Sub(lhs, rhs);
    case Expr::Kind::kMul:
      return Mul(lhs, rhs);
    case Expr::Kind::kDiv:
      return Div(lhs, rhs);
    case Expr::Kind::kConstant:
    case Expr::Kind::kVariable:
      return expr;
  }
}

std::optional<int64_t> Evaluate(const ExprHandle& expr) {
  if (!expr) {
    return std::nullopt;
  }
  if (expr->is_constant()) {
    return expr->constant_value();
  }
  if (expr->is_variable()) {
    return std::nullopt;
  }

  std::optional<int64_t> lhs = Evaluate(expr->lhs());
  std::optional<int64_t> rhs = Evaluate(expr->rhs());
  if (!lhs.has_value() || !rhs.has_value()) {
    return std::nullopt;
  }

  switch (expr->kind()) {
    case Expr::Kind::kAdd:
      return *lhs + *rhs;
    case Expr::Kind::kSub:
      return *lhs - *rhs;
    case Expr::Kind::kMul:
      return *lhs * *rhs;
    case Expr::Kind::kDiv:
      if (*rhs == 0) {
        return std::nullopt;
      }
      return *lhs / *rhs;
    case Expr::Kind::kConstant:
      return expr->constant_value();
    case Expr::Kind::kVariable:
      return std::nullopt;
  }
}

std::string ToString(const ExprHandle& expr) { return ToStringImpl(expr); }

}  // namespace xla::symbolic_expr