from copy import deepcopy
from collections import deque

import sympy as sp
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .MultKAN import *
from .MultKAN import MultKAN as MultKAN_raw

"""
Customization by Damdae Park
"""


def detach(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    elif isinstance(value, (tuple, list)):
        return type(value)(detach(v) for v in value)
    else:
        return value


def replace_half_powers(expr):
    half = sp.Rational(1, 2)
    new_expr = deepcopy(expr)
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Pow):
            exp = sub.exp
            if (exp == half 
                or (exp.is_Float and abs(float(exp) - 0.5) < 1e-14) 
                or exp.equals(half)):
                new_expr = new_expr.xreplace({sub: sp.sqrt(sub.base)})
    return new_expr


def prune_small_terms(expr, eps=1e-5):
    expr = sp.expand(expr)
    terms = expr.as_ordered_terms()
    filtered_terms = [t for t in terms if abs(t.as_coeff_Mul()[0]) > eps]
    return sum(filtered_terms)


def truncate_numbers(expr: sp.Expr, lo=1e-3, hi=1e3, tiny=1e-15, tol_int=5e-4, tol_pow=1e-12) -> sp.Expr:  # added by DP
    """
    Return a new SymPy expression where all numeric atoms are replaced by numbers
    formatted per rules:

      - |a| < tiny              -> exact 0 (Integer)
      - |a| < lo or |a| >= hi   -> scientific, 2 decimals (Float)
      - lo <= |a| < 1           -> fixed, 3 decimals (Float)
      - 1 <= |a| < 10           -> fixed, 2 decimals (Float)
      - 10 <= |a| < 100         -> fixed, 1 decimal  (Float)
      - 100 <= |a| < hi         -> integer (rounded, Integer)

    Extras:
      - Near-integers snap to Integer if |a - round(a)| < tol_int and 1 <= |round(a)| < hi
      - Powers: if exponent ~ integer within tol_pow, snap to that integer; if 1, drop '**1'
    """

    # ---- number formatter with the new bins ----
    def _format_number(a: sp.Number):
        val = float(a)
        aval = abs(val)

        # exact zero
        if aval < tiny:
            return sp.Integer(0)

        # near-integer snap (for clean coefficients)
        r = round(val)
        if abs(val - r) < tol_int and 1 <= abs(r) < hi:
            return sp.Integer(int(r))

        # magnitude-based formatting (bins updated with 10 and 100)
        if aval < lo or aval >= hi:
            return sp.Float(f"{val:.2e}")     # scientific
        elif aval < 1:
            return sp.Float(f"{val:.3f}")     # 0.xxx
        elif aval < 10:
            return sp.Float(f"{val:.2f}")     # xx.xx
        elif aval < 100:
            return sp.Float(f"{val:.1f}")     # xx.x
        else:
            # 100 <= |a| < hi  (i.e., < 1e3 by default)
            # integer rounding with sign preserved
            return sp.Integer(int(r) if val >= 0 else -int(-r))

    out = expr

    # apply to all numeric atoms globally (no identity dependence)
    out = out.replace(lambda a: isinstance(a, sp.Number), _format_number)

    # ---- tidy powers: **1 -> base, **k.0 -> **k if k ~ integer ----
    def _fix_pow(p: sp.Pow):
        base, exp = p.as_base_exp()
        if exp.is_Number:
            ev = float(exp)
            rv = round(ev)
            if abs(ev - rv) < tol_pow:
                if rv == 1:
                    return base
                return base**int(rv)
        return p

    out = out.replace(lambda p: isinstance(p, sp.Pow) and p.exp.is_Number, _fix_pow)
    return out



class MultKAN(MultKAN_raw):
    def __init__(self,
                 width=None, grid=3, k=3, mult_arity=2, noise_scale=0.3,
                 scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu',
                 symbolic_enabled=True, affine_trainable=False, grid_eps=1.0,  # changed grid_eps value from 0.02 to 1.0
                 grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, seed=1,
                 save_act=True, sparse_init=False, auto_save=False,  # changed auto_save from True to False
                 first_init=True, ckpt_path='./model', state_id=0, round=0, device='cpu'):
        super().__init__(width=width, grid=grid, k=k, mult_arity=mult_arity,
                         noise_scale=noise_scale, scale_base_mu=scale_base_mu,
                         scale_base_sigma=scale_base_sigma, base_fun=base_fun,
                         symbolic_enabled=symbolic_enabled, affine_trainable=affine_trainable,
                         grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable,
                         sb_trainable=sb_trainable, seed=seed, save_act=save_act,
                         sparse_init=sparse_init, auto_save=auto_save,
                         first_init=first_init, ckpt_path=ckpt_path, state_id=state_id,
                         round=round, device=device)
        
    def log_history(self, method_name, verbose=True):  # added verbose=True
        if self.auto_save:
            # save to log file
            #print(func.__name__)
            with open(self.ckpt_path+'/history.txt', 'a') as file:
                file.write(str(self.round) + '.' + str(self.state_id)+' => '+ method_name + ' => ' 
                           + str(self.round) + '.' + str(self.state_id+1) + '\n')

            # update state_id
            self.state_id += 1

            # save to ckpt
            self.saveckpt(path=self.ckpt_path + '/' + str(self.round) + '.' + str(self.state_id))
            
            if verbose:
                print('saving model version ' + str(self.round) + '.' + str(self.state_id))


    def refine(self, new_grid, log_history=False):  # added log_history=False
        model_new = self.__class__(width=self.width,
                                   grid=new_grid,
                                   k=self.k,
                                   mult_arity=self.mult_arity,
                                   base_fun=self.base_fun_name,
                                   symbolic_enabled=self.symbolic_enabled,
                                   affine_trainable=self.affine_trainable,
                                   grid_eps=self.grid_eps,
                                   grid_range=self.grid_range,
                                   sp_trainable=self.sp_trainable,
                                   sb_trainable=self.sb_trainable,
                                   ckpt_path=self.ckpt_path,
                                   auto_save=True,
                                   first_init=False,
                                   state_id=self.state_id,
                                   round=self.round,
                                   device=self.device)

        model_new.initialize_from_another_model(self, self.cache_data)
        model_new.cache_data = self.cache_data
        model_new.grid = new_grid
        
        if log_history:
            self.log_history('refine')
        model_new.state_id += 1
        
        return model_new.to(self.device)
    
    
    def forward(self, x, singularity_avoiding=True, y_th=10.0):  # changed singularity_avoiding from False to True
        '''
        forward pass
        
        Args:
        -----
            x : 2D torch.tensor
                inputs
            singularity_avoiding : bool
                whether to avoid singularity for the symbolic branch
            y_th : float
                the threshold for singularity

        Returns:
        --------
            None
            
        Example1
        --------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> x = torch.rand(100,2)
        >>> model(x).shape
        
        Example2
        --------
        >>> from kan import *
        >>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
        >>> x = torch.tensor([[1],[-0.01]])
        >>> model.fix_symbolic(0,0,0,'log',fit_params_bool=False)
        >>> print(model(x))
        >>> print(model(x, singularity_avoiding=True))
        >>> print(model(x, singularity_avoiding=True, y_th=1.))
        '''
        x = x[:,self.input_id.long()]
        assert x.shape[1] == self.width_in[0]
        
        # cache data
        self.cache_data = x
        
        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
        self.subnode_actscale = []
        self.edge_actscale = []
        # self.neurons_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):
            
            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)
            #print(preacts, postacts_numerical, postspline)
            
            if torch.isnan(x_numerical).any():  # added checker
                raise ValueError(f"NaN in x_numerical at layer {l}")
            elif torch.isnan(preacts).any():
                raise ValueError(f"NaN in preacts at layer {l}")
            elif torch.isnan(postacts_numerical).any():
                raise ValueError(f"NaN in postacts_numerical at layer {l}")
            elif torch.isnan(postspline).any():
                raise ValueError(f"NaN in postspline at layer {l}")
            else:
                pass
            
            if self.symbolic_enabled == True:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x, singularity_avoiding=singularity_avoiding, y_th=y_th)
            else:
                x_symbolic = 0.
                postacts_symbolic = 0.

            x = x_numerical + x_symbolic
            
            if self.save_act:
                # save subnode_scale
                self.subnode_actscale.append(torch.std(x, dim=0).detach())
            
            # subnode affine transform
            x = self.subnode_scale[l][None,:] * x + self.subnode_bias[l][None,:]
            
            if self.save_act:
                postacts = postacts_numerical + postacts_symbolic

                # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
                #grid_reshape = self.act_fun[l].grid.reshape(self.width_out[l + 1], self.width_in[l], -1)
                input_range = torch.std(preacts, dim=0) + 0.1
                output_range_spline = torch.std(postacts_numerical, dim=0) # for training, only penalize the spline part
                output_range = torch.std(postacts, dim=0) # for visualization, include the contribution from both spline + symbolic
                # save edge_scale
                self.edge_actscale.append(output_range)
                
                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_spline.append(output_range_spline / input_range)
                self.spline_preacts.append(preacts.detach())
                self.spline_postacts.append(postacts.detach())
                self.spline_postsplines.append(postspline.detach())

                self.acts_premult.append(x.detach())
            
            # multiplication
            dim_sum = self.width[l+1][0]
            dim_mult = self.width[l+1][1]
            
            if self.mult_homo == True:
                for i in range(self.mult_arity-1):
                    if i == 0:
                        x_mult = x[:,dim_sum::self.mult_arity] * x[:,dim_sum+1::self.mult_arity]
                    else:
                        x_mult = x_mult * x[:,dim_sum+i+1::self.mult_arity]
                        
            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l+1][:j])
                    for i in range(self.mult_arity[l+1][j]-1):
                        if i == 0:
                            x_mult_j = x[:,[acml_id]] * x[:,[acml_id+1]]
                        else:
                            x_mult_j = x_mult_j * x[:,[acml_id+i+1]]
                            
                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)
                
            if self.width[l+1][1] > 0:
                x = torch.cat([x[:,:dim_sum], x_mult], dim=1)
            
            # x = x + self.biases[l].weight
            # node affine transform
            x = self.node_scale[l][None,:] * x + self.node_bias[l][None,:]
            
            self.acts.append(x.detach())
            
        
        return x
    
    
    def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10),
                     verbose=True, random=False, log_history=False):  # changed log_history from True to False
        r2 = super().fix_symbolic(l, i, j, fun_name, fit_params_bool=fit_params_bool, 
                                  a_range=a_range, b_range=b_range, verbose=verbose,
                                  random=random, log_history=log_history)
        if log_history:
            self.log_history('fix_symbolic', verbose=verbose)
        
        try:
            return r2
        except NameError:
            return None


    def unfix_symbolic(self, l, i, j, log_history=False):  # changed log_history from True to False
        return super().unfix_symbolic(l, i, j, log_history=log_history)


    def unfix_symbolic_all(self, log_history=False):  # changed log_history from True to False
        return super().unfix_symbolic_all(log_history=log_history)
    
    
    def loadckpt(path='model'):
        instance = super().load_ckpt(path=path)
        return self.wrapping_methods(instance)
    
    def rewind(self, model_id):
        instance = super().rewind(model_id)
        return self.wrapping_methods(instance)
    
    def checkout(self, model_id):
        instance = super().checkout(model_id)
        return self.wrapping_methods(instance)
    
    
    def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1.,
            lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True,
            grid_update_num=10, loss_fn=None, lr=1., start_grid_update_step=-1,
            stop_grid_update_step=50, batch=-1, metrics=None, save_fig=False,
            in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video',
            singularity_avoiding=True, y_th=1000., reg_metric='edge_forward_spline_n',
            display_metrics=None, verbose=True, log_history=False, shuffle=False,
            eval_test_loss=True, early_stop=True, es_patience=30, es_min_delta=0.0, 
            es_rel_delta=0.0, es_monitor='test_rmse', clip_grad_norm=1.0, clip_grad_value=None):  
        # added verbose, eval_test_loss, early_stop related parameters
        
        '''
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            start_grid_update_step : int
                no grid updates before this training step
            stop_grid_update_step : int
                no grid updates after this training step
            loss_fn : function
                loss function
            lr : float
                learning rate
            batch : int
                batch size, if -1 then full.
            save_fig_freq : int
                save figure every (save_fig_freq) steps
            singularity_avoiding : bool
                indicate whether to avoid singularity for the symbolic part
            y_th : float
                singularity threshold (anything above the threshold is considered singular and is softened in some ways)
            reg_metric : str
                regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
            metrics : a list of metrics (as functions)
                the metrics to be computed in training
            display_metrics : a list of functions
                the metric to be displayed in tqdm bar
            verbose : bool
                whether to display training process in tqdm bar
            eval_test_loss : bool
                whether to evaluate test loss
            early_stop:
                If True, stop when the monitored metric does not improve for es_patience steps.
            es_patience:
                Number of steps to wait for an improvement before stopping.
            es_min_delta:
                Minimum absolute improvement required to reset patience.
            es_rel_delta:
                Relative improvement required to reset patience.
            es_monitor:
                One of {'test_rmse','train_rmse','objective'}.
                - 'test_rmse': sqrt(test_loss) if eval_test_loss=True, else falls back to 'objective'.
                - 'train_rmse': sqrt(train_loss).
                - 'objective': train_loss + lamb * reg_.
            clip_grad_norm : float or None
                Maximum L2 norm of the gradients. If None, no clipping is applied. Default: 1.0.
            clip_grad_value : float or None
                Maximum absolute value of the gradients. If None, no clipping is applied. Default: None.
        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
                other metrics specified in metrics

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        # Most examples in toturals involve the fit() method. Please check them for useness.
        '''

        if lamb > 0. and not self.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')
            
        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        if verbose:
            pbar = tqdm(range(steps), desc='description', ncols=100)
        else:
            pbar = range(steps)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "AdamW":  # scheduling applied for better convergence; added
            bounds = (-10, 10)
            for name, weights in self.named_parameters():
                if weights.requires_grad:
                    weights.register_hook(lambda grad, bounds=bounds: torch.clamp(grad, bounds[0], bounds[1]))
                    
            optimizer = torch.optim.AdamW(self.get_params(), lr=lr)
            scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.8, min_lr=1e-6)
        elif opt == "Adam":
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.get_params(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            try:
                pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
            except ValueError:
                # NaN detected in forward pass — return large finite loss so
                # LBFGS can backtrack to a smaller step size gracefully.
                train_loss = torch.tensor(1e6, requires_grad=True)
                reg_ = torch.tensor(0.)
                return train_loss
            train_loss = loss_fn(pred, dataset['train_label'][train_id])
            if self.save_act:
                if reg_metric == 'edge_backward':
                    self.attribute()
                if reg_metric == 'node_backward':
                    self.node_attribute()
                reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                reg_ = torch.tensor(0.)
            objective = train_loss + lamb * reg_
            objective.backward()
            # gradient clipping for LBFGS
            if clip_grad_norm is not None or clip_grad_value is not None:
                try:
                    params = list(self.get_params())
                except Exception:
                    params = list(self.parameters())
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, clip_grad_norm)
                if clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(params, clip_grad_value)
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        # Early-stopping state
        best_metric = float('inf')
        es_counter = 0

        # Smoothing/window settings
        ema_metric = None
        ema_alpha = 0.2           # lower -> smoother; tune 0.1~0.3
        window = 20               # compare against best in the last 'window' steps
        metric_hist = deque(maxlen=window)

        # Controls
        grace_steps = 50          # do not early-stop before this many steps
        use_relative = True       # use relative delta along with absolute delta
        
        for i in pbar:
            if i == steps-1 and old_save_act:
                self.save_act = True
                
            if save_fig and i % save_fig_freq == 0:
                save_act = self.save_act
                self.save_act = True
            
            if shuffle:
                train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
                test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)
            else:
                train_id = list(range(batch_size))
                test_id = list(range(batch_size_test))

            if i % grid_update_freq == 0 and i < stop_grid_update_step and update_grid and i >= start_grid_update_step:
                self.update_grid(dataset['train_input'][train_id])

            if opt in ["AdamW", "Adam"]:  # edited
                try:
                    pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
                    train_loss = loss_fn(pred, dataset['train_label'][train_id])
                except ValueError:
                    train_loss = torch.tensor(1e6, requires_grad=True)
                    reg_ = torch.tensor(0.)
                    loss = train_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    continue
                if self.save_act:
                    if reg_metric == 'edge_backward':
                        self.attribute()
                    if reg_metric == 'node_backward':
                        self.node_attribute()
                    reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping for Adam/AdamW
                if clip_grad_norm is not None or clip_grad_value is not None:
                    try:
                        params = list(self.get_params())
                    except Exception:
                        params = list(self.parameters())
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params, clip_grad_norm)
                    if clip_grad_value is not None:
                        torch.nn.utils.clip_grad_value_(params, clip_grad_value)
                optimizer.step()
                
                if opt == "AdamW":  # added
                    metric = test_loss if eval_test_loss else loss
                    scheduler.step(metric)
                    
            elif opt == "LBFGS":
                optimizer.step(closure)

            if eval_test_loss:  # added
                try:
                    test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])
                except ValueError:
                    test_loss = torch.tensor(1e6)
            else:
                test_loss = torch.Tensor(0.0)  # dummy
            
            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if es_monitor == 'test_rmse' and eval_test_loss:
                raw_metric = torch.sqrt(test_loss).detach().cpu().item()
            elif es_monitor == 'train_rmse':
                raw_metric = torch.sqrt(train_loss).detach().cpu().item()
            else:
                raw_metric = (train_loss + lamb * reg_).detach().cpu().item()

            # apply EMA smoothing
            ema_metric = raw_metric if ema_metric is None else (ema_alpha * raw_metric + (1 - ema_alpha) * ema_metric)

            # keep windowed history (store smoothed values)
            metric_hist.append(ema_metric)

            # compute reference: best so far AND best-in-window (robust to brief spikes)
            ref_global = best_metric
            ref_window = np.min(metric_hist) if len(metric_hist) > 0 else ema_metric
            ref = min(ref_global, ref_window)

            # compute improvement thresholds
            abs_thresh = es_min_delta
            rel_thresh = es_rel_delta * abs(ref) if use_relative else 0.0
            need_improve = max(abs_thresh, rel_thresh)

            # decide improvement only after grace_steps
            if early_stop and len(metric_hist) >= min(grace_steps, window):
                if ref - ema_metric > need_improve:
                    # improvement: reset counter and update global best
                    best_metric = min(best_metric, ema_metric)
                    es_counter = 0
                else:
                    # no improvement
                    es_counter += 1
                    if es_counter >= es_patience:
                        break
            
            if verbose and i % log == 0:  # added monitoring parameter
                if display_metrics == None:
                    pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)
                    
            if save_fig and i % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(i), beta=beta)
                plt.savefig(img_folder + '/' + str(i) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()
                self.save_act = save_act
            
            
        if log_history:
            self.log_history('fit')
            
        # revert back to original state
        self.symbolic_enabled = old_symbolic_enabled
        return results
    
    
    def wrapping_methods(self, instance):
        instance.__class__ = self.__class__  # transform parent instance to child instance
        return instance
    
    
    def prune_node(self, threshold=1e-2, mode="auto", active_neurons_id=None, log_history=False):
        """
        Modified to ensure leaving at least one node
        """
        if self.acts is None:
            self.get_act()
        
        # masks and active sets
        mask_up = [torch.ones(self.width_in[0], device=self.device)]
        mask_down = []
        active_neurons_up = [list(range(self.width_in[0]))]
        active_neurons_down = []
        num_sums = []
        num_mults = []
        mult_arities = [[]]

        if active_neurons_id is not None:
            mode = 'manual'

        for i in range(len(self.acts_scale) - 1):
            # layer index i: mapping from in_dim at i+1 to down nodes at i
            mult_arity = []

            if mode == 'auto':
                # compute attribution once per layer
                self.attribute()
                scores_up = self.node_scores[i+1]
                overall_important_up = scores_up > threshold

            elif mode == 'manual':
                # manual selection corresponds to the inputs of layer i+1
                overall_important_up = torch.zeros(self.width_in[i + 1], dtype=torch.bool, device=self.device)
                sel = active_neurons_id[i+1] if (isinstance(active_neurons_id, (list, tuple)) and len(active_neurons_id) > i+1) else []
                if len(sel) > 0:
                    overall_important_up[torch.as_tensor(sel, device=self.device, dtype=torch.long)] = True

            # ensure at least one upstream node remains
            if overall_important_up.sum() == 0:
                if mode == 'auto':
                    keep = int(torch.argmax(self.node_scores[i+1]).item())
                else:
                    keep = 0
                overall_important_up[keep] = True

            # map upstream mask to downstream subnodes
            num_sum = torch.sum(overall_important_up[:self.width[i+1][0]])
            num_mult = torch.sum(overall_important_up[self.width[i+1][0]:])

            if self.mult_homo:
                base = overall_important_up[:self.width[i+1][0]]
                tail = overall_important_up[self.width[i+1][0]:]
                if tail.numel() > 0:
                    tiled = (tail[None, :].expand(self.mult_arity, -1)).T.reshape(-1,)
                else:
                    tiled = torch.zeros(0, dtype=torch.bool, device=self.device)
                overall_important_down = torch.cat([base, tiled], dim=0)
            else:
                overall_important_down = overall_important_up[:self.width[i+1][0]].clone()
                tail = overall_important_up[self.width[i+1][0]:]
                for j in range(tail.shape[0]):
                    active_bool = bool(tail[j].item())
                    arity = self.mult_arity[i+1][j]
                    overall_important_down = torch.cat([overall_important_down, torch.tensor([active_bool]*arity, device=self.device)])
                    if active_bool:
                        mult_arity.append(arity)

            # ensure at least one downstream subnode remains
            if overall_important_down.sum() == 0:
                # keep the most important upstream and its first mapped downstream
                keep_up = int(torch.argmax(self.node_scores[i+1]).item()) if mode == 'auto' else 0
                overall_important_up[:] = False
                overall_important_up[keep_up] = True
                # rebuild downstream mapping minimally
                if self.mult_homo:
                    base = overall_important_up[:self.width[i+1][0]]
                    tail = overall_important_up[self.width[i+1][0]:]
                    if tail.numel() > 0:
                        tiled = (tail[None, :].expand(self.mult_arity, -1)).T.reshape(-1,)
                    else:
                        tiled = torch.zeros(0, dtype=torch.bool, device=self.device)
                    overall_important_down = torch.cat([base, tiled], dim=0)
                else:
                    overall_important_down = overall_important_up[:self.width[i+1][0]].clone()
                    tail = overall_important_up[self.width[i+1][0]:]
                    mult_arity = []
                    for j in range(tail.shape[0]):
                        active_bool = bool(tail[j].item())
                        arity = self.mult_arity[i+1][j]
                        overall_important_down = torch.cat([overall_important_down, torch.tensor([active_bool]*arity, device=self.device)])
                        if active_bool:
                            mult_arity.append(arity)
                # if still all False due to structure, force the very first downstream on
                if overall_important_down.sum() == 0 and overall_important_down.numel() > 0:
                    overall_important_down[0] = True
                    if num_sum == 0 and self.width[i+1][0] > 0:
                        num_sum = 1

            num_sums.append(int(num_sum.item()))
            num_mults.append(int(num_mult.item()))

            mask_up.append(overall_important_up.float())
            mask_down.append(overall_important_down.float())

            active_neurons_up.append(torch.where(overall_important_up)[0].tolist())
            active_neurons_down.append(torch.where(overall_important_down)[0].tolist())

            mult_arities.append(mult_arity)

        # keep all outputs by default; can be pruned later by edge pruning safely
        active_neurons_down.append(list(range(self.width_out[-1])))
        mask_down.append(torch.ones(self.width_out[-1], device=self.device))

        if not self.mult_homo:
            mult_arities.append(self.mult_arity[-1])

        self.mask_up = mask_up
        self.mask_down = mask_down

        # apply node removals based on active sets
        for l in range(len(self.acts_scale) - 1):
            # remove inputs (up) at layer l+1
            keep_up = set(active_neurons_up[l + 1])
            for i in range(self.width_in[l + 1]):
                if i not in keep_up:
                    self.remove_node(l + 1, i, mode='up', log_history=False)
            # remove outputs (down) at layer l
            keep_down = set(active_neurons_down[l])
            for i in range(self.width_out[l + 1]):
                if i not in keep_down:
                    self.remove_node(l + 1, i, mode='down', log_history=False)

        model2 = MultKAN(
            copy.deepcopy(self.width),
            grid=self.grid, k=self.k,
            base_fun=self.base_fun_name,
            mult_arity=self.mult_arity,
            ckpt_path=self.ckpt_path,
            auto_save=True, first_init=False,
            state_id=self.state_id, round=self.round
        ).to(self.device)
        model2.load_state_dict(self.state_dict())

        width_new = [self.width[0]]

        for i in range(len(self.acts_scale)):
            if i < len(self.acts_scale) - 1:
                num_sum = num_sums[i]
                num_mult = num_mults[i]

                up_idx = active_neurons_up[i+1]
                down_idx = active_neurons_down[i]

                # shrink node params
                model2.node_bias[i].data = model2.node_bias[i].data[up_idx]
                model2.node_scale[i].data = model2.node_scale[i].data[up_idx]
                model2.subnode_bias[i].data = model2.subnode_bias[i].data[down_idx]
                model2.subnode_scale[i].data = model2.subnode_scale[i].data[down_idx]

                # update width bookkeeping
                model2.width[i+1] = [num_sum, num_mult]
                model2.act_fun[i].out_dim_sum = num_sum
                model2.act_fun[i].out_dim_mult = num_mult
                model2.symbolic_fun[i].out_dim_sum = num_sum
                model2.symbolic_fun[i].out_dim_mult = num_mult

                width_new.append([num_sum, num_mult])

            # slice layers to active subsets
            model2.act_fun[i] = model2.act_fun[i].get_subset(active_neurons_up[i], active_neurons_down[i])
            model2.symbolic_fun[i] = self.symbolic_fun[i].get_subset(active_neurons_up[i], active_neurons_down[i])

        model2.cache_data = self.cache_data
        model2.acts = None

        width_new.append(self.width[-1])
        model2.width = width_new

        if not self.mult_homo:
            model2.mult_arity = mult_arities

        if log_history:
            self.log_history('prune_node')
            model2.state_id += 1

        return model2
    
    
    def prune_edge(self, threshold=3e-2, log_history=False):
        """
        pruning edges while leaving at least one edge
        """
        if self.acts is None:
            self.get_act()

        for i in range(len(self.width) - 1):
            # keep original orientation: scores [out, in], mask used with permute(1,0)
            scores_oi = self.edge_scores[i]                  # [out, in]
            old_mask_io = self.act_fun[i].mask.data          # expected [in, out]
            cand_io = (scores_oi > threshold).permute(1, 0)  # [in, out]
            new_mask_io = (cand_io * old_mask_io).float()

            # layer-level guarantee: ensure at least one edge remains
            if new_mask_io.numel() > 0 and new_mask_io.sum() == 0:
                # prefer best position allowed by old_mask; fallback to global best
                allowed_oi = old_mask_io.permute(1, 0)       # [out, in]
                if allowed_oi.sum() > 0:
                    scores_allowed = scores_oi * allowed_oi
                    flat = scores_allowed.view(-1)
                    idx = int(flat.argmax().item())
                    out_j = idx // scores_oi.shape[1]
                    in_i = idx % scores_oi.shape[1]
                    new_mask_io[in_i, out_j] = 1.0
                else:
                    # very defensive fallback if nothing is allowed but we must keep one
                    flat = scores_oi.view(-1)
                    idx = int(flat.argmax().item())
                    out_j = idx // scores_oi.shape[1]
                    in_i = idx % scores_oi.shape[1]
                    new_mask_io[in_i, out_j] = 1.0

            # node-level guarantee: avoid isolated outputs and inputs
            if new_mask_io.numel() > 0:
                # fix outputs with no incoming edges
                for out_j in range(new_mask_io.shape[1]):
                    if new_mask_io[:, out_j].sum() == 0:
                        col_allowed_in = old_mask_io[:, out_j]  # [in]
                        if col_allowed_in.sum() > 0:
                            # choose best incoming among allowed
                            in_scores = scores_oi[out_j] * col_allowed_in
                            in_i = int(in_scores.argmax().item())
                            new_mask_io[in_i, out_j] = 1.0
                        else:
                            # fallback to best incoming regardless
                            in_i = int(scores_oi[out_j].argmax().item())
                            new_mask_io[in_i, out_j] = 1.0

                # fix inputs with no outgoing edges
                for in_i in range(new_mask_io.shape[0]):
                    if new_mask_io[in_i, :].sum() == 0:
                        row_allowed_out = old_mask_io[in_i, :]   # [out]
                        if row_allowed_out.sum() > 0:
                            out_scores = scores_oi[:, in_i] * row_allowed_out
                            out_j = int(out_scores.argmax().item())
                            new_mask_io[in_i, out_j] = 1.0
                        else:
                            out_j = int(scores_oi[:, in_i].argmax().item())
                            new_mask_io[in_i, out_j] = 1.0

            self.act_fun[i].mask.data = new_mask_io

        if log_history:
            self.log_history('prune_edge')
    
    
    def prune(self, node_th=1e-2, edge_th=3e-2, log_history=False, use_parent_func=False):
        if self.acts == None:
            self.get_act()
        
        if use_parent_func:
            self = super().prune_node(node_th, log_history=False)
        else:
            self = self.prune_node(node_th, log_history=False)
            
        self.forward(self.cache_data)
        self.attribute()
        
        if use_parent_func:
            super().prune_edge(edge_th, log_history=False)
        else:
            self.prune_edge(edge_th, log_history=False)
        
        if log_history:
            self.log_history('prune')
        return self
    
    
    def prune_input(self, threshold=1e-2, active_inputs=None, log_history=False):
        '''
        prune inputs

        Args:
        -----
            threshold : float
                if the attribution score of the input feature is below threshold, it is considered irrelevant.
            active_inputs : None or list
                if a list is passed, the manual mode will disregard attribution score and prune as instructed.
            
        Returns:
        --------
            pruned network : MultKAN

        Example1
        --------
        >>> # automatic
        >>> from kan import *
        >>> model = KAN(width=[3,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: 1 * x[:,[0]]**2 + 0.3 * x[:,[1]]**2 + 0.0 * x[:,[2]]**2
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        >>> model = model.prune_input()
        >>> model.plot()
        
        Example2
        --------
        >>> # automatic
        >>> from kan import *
        >>> model = KAN(width=[3,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: 1 * x[:,[0]]**2 + 0.3 * x[:,[1]]**2 + 0.0 * x[:,[2]]**2
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        >>> model = model.prune_input(active_inputs=[0,1])
        >>> model.plot()
        '''
        if active_inputs == None:
            self.attribute()
            input_score = self.node_scores[0]
            input_mask = input_score > threshold
            print('keep:', input_mask.tolist())
            input_id = torch.where(input_mask==True)[0]
            
        else:
            input_id = torch.tensor(active_inputs, dtype=torch.long).to(self.device)
        
        model2 = MultKAN(copy.deepcopy(self.width), grid=self.grid, k=self.k, base_fun=self.base_fun, mult_arity=self.mult_arity, ckpt_path=self.ckpt_path, auto_save=True, first_init=False, state_id=self.state_id, round=self.round).to(self.device)
        model2.load_state_dict(self.state_dict())

        model2.act_fun[0] = model2.act_fun[0].get_subset(input_id, torch.arange(self.width_out[1]))
        model2.symbolic_fun[0] = self.symbolic_fun[0].get_subset(input_id, torch.arange(self.width_out[1]))

        model2.cache_data = self.cache_data
        model2.acts = None

        model2.width[0] = [len(input_id), 0]
        model2.input_id = input_id
        
        if log_history:
            self.log_history('prune_input')
        
        model2.state_id += 1  # unindented
        return model2
    

    def remove_edge(self, l, i, j, log_history=False):
        return super().remove_edge(l, i, j, log_history=log_history)
    
    
    def remove_node(self, l, i, mode='all', log_history=False):
        return super().remove_node(l, i, mode=mode, log_history=log_history)
    
    
    def auto_symbolic(self, a_range=(-30, 30), b_range=(-30, 30), lib=None,  # increased a_range and b_range spans
                      verbose=1, weight_simple=0.8, r2_threshold=0.0, log_history=False):
        '''
        automatic symbolic regression for all edges

        Args:
        -----
            a_range : tuple
                search range of a
            b_range : tuple
                search range of b
            lib : list of str
                library of candidate symbolic functions
            verbose : int
                larger verbosity => more verbosity
            weight_simple : float
                a weight that prioritizies simplicity (low complexity) over performance (high r2) - set to 0.0 to ignore complexity
            r2_threshold : float
                If r2 is below this threshold, the edge will not be fixed with any symbolic function - set to 0.0 to ignore this threshold
        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,1,1], grid=5, k=3, noise_scale=0.0, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.auto_symbolic()
        '''
        
        for l in range(len(self.width_in) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    if self.symbolic_fun[l].mask[j, i] > 0. and self.act_fun[l].mask[i][j] == 0.:
                        if verbose >= 1:
                            print(f'skipping ({l},{i},{j}) since already symbolic')
                    elif self.symbolic_fun[l].mask[j, i] == 0. and self.act_fun[l].mask[i][j] == 0.:
                        self.fix_symbolic(l, i, j, '0', verbose=verbose, log_history=False)
                        if verbose >= 1:
                            print(f'fixing ({l},{i},{j}) with 0')
                    else:
                        name, fun, r2, c = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=lib, verbose=False, weight_simple=weight_simple)
                        
                        if r2 >= r2_threshold:
                            self.fix_symbolic(l, i, j, name, verbose=verbose, log_history=False)
                            if verbose >= 1:
                                print(f'fixing ({l},{i},{j}) with {name}, r2={round(r2, 2)}, c={c}')
                        else:
                            if verbose >= 1:
                                print(f'For ({l},{i},{j}) the best fit was {name}, but r^2 = {r2} and this is lower than {r2_threshold}. This edge was omitted, keep training or try a different threshold.')
        
        if log_history:
            self.log_history('auto_symbolic', verbose=verbose)
    

    def symbolic_formula(self, var=None, normalizer=None, output_normalizer=None, simplify=False):
        '''
        get symbolic formula

        Args:
        -----
            var : None or a list of sp expression
                input variables
            normalizer : [mean, std]
            output_normalizer : [mean, std]
            
        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,1,1], grid=5, k=3, noise_scale=0.0, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.auto_symbolic()
        >>> model.symbolic_formula()[0][0]
        '''
        
        symbolic_acts = []
        symbolic_acts_premult = []
        x = []

        def ex_round(ex1, n_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, n_digit))
            return ex2

        # define variables
        if var == None:
            for ii in range(1, self.width[0][0] + 1):
                exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                exec(f"x.append(x{ii})")
        elif isinstance(var[0], sympy.Expr):
            x = var
        else:
            x = [sympy.symbols(var_) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = normalizer[0]
            std = normalizer[1]
            x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        for l in range(len(self.width_in) - 1):
            num_sum = self.width[l + 1][0]
            num_mult = self.width[l + 1][1]
            y = []
            for j in range(self.width_out[l + 1]):
                yj = 0.
                for i in range(self.width_in[l]):
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    
                    (a,b,c,d) = detach((a, b, c, d))
                    
                    try:
                        yj += c * sympy_fun(a * x[i] + b) + d  # here
                    except:
                        print('make sure all activations need to be converted to symbolic formulas first!')
                        return
                yj = self.subnode_scale[l][j] * yj + self.subnode_bias[l][j]
                
                if simplify == True:
                    y.append(sympy.simplify(yj))  #! takes too long
                else:
                    y.append(yj)
                    
            symbolic_acts_premult.append(y)
                  
            mult = []
            for k in range(num_mult):
                if isinstance(self.mult_arity, int):
                    mult_arity = self.mult_arity
                else:
                    mult_arity = self.mult_arity[l+1][k]
                for i in range(mult_arity-1):
                    if i == 0:
                        mult_k = y[num_sum+2*k] * y[num_sum+2*k+1]
                    else:
                        mult_k = mult_k * y[num_sum+2*k+i+1]
                mult.append(mult_k)
                
            y = y[:num_sum] + mult
            
            for j in range(self.width_in[l+1]):
                y[j] = self.node_scale[l][j] * y[j] + self.node_bias[l][j]
            
            x = y
            symbolic_acts.append(x)

        if output_normalizer != None:
            output_layer = symbolic_acts[-1]
            means = output_normalizer[0]
            stds = output_normalizer[1]

            assert len(output_layer) == len(means), 'output_normalizer does not match the output layer'
            assert len(output_layer) == len(stds), 'output_normalizer does not match the output layer'
            
            output_layer = [(output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))]
            symbolic_acts[-1] = output_layer


        self.symbolic_acts = [[symbolic_acts[l][i] for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]
        self.symbolic_acts_premult = [[symbolic_acts_premult[l][i] for i in range(len(symbolic_acts_premult[l]))] for l in range(len(symbolic_acts_premult))]

        out_dim = len(symbolic_acts[-1])
        #return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        
        return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
    
    
    def swap(self, l, i1, i2, log_history=False):
        return super().swap(l, i1, i2, log_history=log_history)


    def eval_complexity(self, threshold=0.0):  # added
        complexity = 0
        for l in range(len(self.width_in) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    num_active = (self.act_fun[l].mask[i][j] > threshold)
                    sym_active = (
                        (self.symbolic_fun[l].mask[j, i] > threshold) and
                        (self.symbolic_fun[l].funs_name[j][i] != "0")
                    )
                    if num_active or sym_active:
                        complexity += 1
        return complexity
