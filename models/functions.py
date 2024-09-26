from torch.autograd import Function

# 旧版梯度反转层
# class ReverseLayerF(Function):

#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha

#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.alpha

#         return output, None
    
# 梯度反转层
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, eta=1.0):
        ctx.eta = eta
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.eta), None


