import jax.numpy as jnp
import jax.lax as lax

from bacadi.utils.func import squared_norm_pytree


class JointAdditiveFrobeniusSEKernel:
    """
    Squared exponential kernel, that simply computes the 
    exponentiated quadratic of the difference in Frobenius norms

    k([Z, theta, interv], [Z', theta', interv]) = 
        scale_z      * exp(- 1/h_z      ||Z - Z'||^2_F)
      + scale_theta  * exp(- 1/h_th     ||theta - theta'||^2_F )
      + scale_interv * exp(- 1/h_interv ||interv - interv'||^2_F )

    """
    def __init__(self,
                 *,
                 h_latent,
                 h_theta,
                 h_interv,
                 scale_latent=1.0,
                 scale_theta=1.0,
                 scale_interv=1.0):
        super().__init__()

        self.h_latent = h_latent
        self.h_theta = h_theta
        self.h_interv = h_interv
        self.scale_latent = scale_latent
        self.scale_theta = scale_theta
        self.scale_interv = scale_interv

    def eval(self,
             *,
             x_latent,
             x_theta,
             x_interv,
             y_latent,
             y_theta,
             y_interv,
             h_latent=-1.0,
             h_theta=-1.0,
             h_interv=-1.0):
        """Evaluates kernel function k(x, y) 
        
        Args:
            x_latent: [...]
            x_theta: PyTree 
            y_latent: [...]
            y_theta: PyTree 
            h_latent: bandwidth for Z term; h_latent == -1 indicates class setting is used
            h_theta: bandwidth for Z term; h_theta == -1 indicates class setting is used
        
        Returns: 
            [1, ]
        """
        # bandwidth (jax-consistent checking which h is used)
        h_latent_ = lax.cond(h_latent == -1.0,
                             lambda _: self.h_latent,
                             lambda _: h_latent,
                             operand=None)

        h_theta_ = lax.cond(h_theta == -1.0,
                            lambda _: self.h_theta,
                            lambda _: h_theta,
                            operand=None)

        h_interv_ = lax.cond(h_interv == -1.0,
                             lambda _: self.h_interv,
                             lambda _: h_interv,
                             operand=None)

        # compute norm
        latent_squared_norm = jnp.sum((x_latent - y_latent)**2.0)
        theta_squared_norm = squared_norm_pytree(x_theta, y_theta)
        interv_squared_norm = jnp.sum((x_interv - y_interv)**2.0)

        # compute kernel
        return (self.scale_latent * jnp.exp(-latent_squared_norm / h_latent_) +
                self.scale_theta * jnp.exp(-theta_squared_norm / h_theta_) +
                self.scale_interv * jnp.exp(-interv_squared_norm / h_interv_))


class MarginalAdditiveFrobeniusSEKernel:
    """
    Squared exponential kernel, that simply computes the 
    exponentiated quadratic of the difference in Frobenius norms

    k([Z, interv], [Z', interv']) = 
        scale_z      * exp(- 1/h_z      ||Z - Z'||^2_F)
      + scale_interv * exp(- 1/h_interv ||interv - interv'||^2_F )
    """
    def __init__(self,
                 *,
                 h_latent,
                 h_interv,
                 scale_latent=1.0,
                 scale_interv=1.0):
        super().__init__()

        self.h_latent = h_latent
        self.h_interv = h_interv
        self.scale_latent = scale_latent
        self.scale_interv = scale_interv

    def eval(self,
             *,
             x_latent,
             x_interv,
             y_latent,
             y_interv,
             h_latent=-1.0,
             h_interv=-1.0):
        """Evaluates kernel function k(x, y) 
        
        Args:
            x_latent: [...]
            x_interv: [...] 
            y_latent: [...]
            y_interv: [...]
            h_latent: bandwidth for Z term; h_latent == -1 indicates class setting is used
            h_interv: bandwidth for interv term; h_interv == -1 indicates class setting is used
        
        Returns: 
            [1, ]
        """
        # bandwidth (jax-consistent checking which h is used)
        h_latent_ = lax.cond(h_latent == -1.0,
                             lambda _: self.h_latent,
                             lambda _: h_latent,
                             operand=None)

        h_interv_ = lax.cond(h_interv == -1.0,
                             lambda _: self.h_interv,
                             lambda _: h_interv,
                             operand=None)

        # compute norm
        latent_squared_norm = jnp.sum((x_latent - y_latent)**2.0)
        interv_squared_norm = jnp.sum((x_interv - y_interv)**2.0)

        # compute kernel
        return (self.scale_latent * jnp.exp(-latent_squared_norm / h_latent_) +
                self.scale_interv * jnp.exp(-interv_squared_norm / h_interv_))
