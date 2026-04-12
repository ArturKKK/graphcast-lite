import torch
import numpy as np

class OptimalInterpolation:
    def __init__(self, grid_lats, grid_lons, sigma_b, sigma_o, L, device,
                 flat_grid=False, roi_idx=None):
        """
        Инициализация Оптимальной Интерполяции.
        sigma_b: ошибка прогноза
        sigma_o: ошибка наблюдений
        L: радиус влияния (в метрах)
        flat_grid: если True, grid_lats/grid_lons — per-node координаты (N,),
                   а не оси для meshgrid.
        roi_idx: если задан (np.array индексов), B строится только для ROI-узлов,
                 а apply() корректирует только их. Нужно для больших сеток (>100K).
        """
        self.sigma_b = sigma_b
        self.sigma_o = sigma_o
        self.L = L
        self.device = device
        self.roi_idx = roi_idx

        if flat_grid:
            # Координаты уже per-node (например, multires flat grid)
            self.grid_coords = np.vstack([grid_lats, grid_lons]).T  # (N, 2)
        else:
            lat_grid, lon_grid = np.meshgrid(grid_lats, grid_lons, indexing='ij')
            self.grid_coords = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T

        if roi_idx is not None:
            # B строим только для ROI-узлов (экономия памяти)
            self._oi_coords = self.grid_coords[roi_idx]
            print(f"[OI] ROI mode: B matrix {len(roi_idx)}×{len(roi_idx)} "
                  f"({len(roi_idx)**2 * 4 / 1e6:.0f} MB)")
        else:
            self._oi_coords = self.grid_coords

        self.B = self._build_B().to(self.device)

    def _dist_matrix(self, coords1, coords2):
        """Считает точное расстояние (в метрах) между точками на сфере."""
        R = 6371000.0
        lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
        lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
        
        dlat = lat1[:, None] - lat2[None, :]
        dlon = lon1[:, None] - lon2[None, :]
        
        a = np.sin(dlat/2)**2 + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def _build_B(self):
        """Строит матрицу ковариации ошибок прогноза."""
        dists = self._dist_matrix(self._oi_coords, self._oi_coords)
        cov = (self.sigma_b**2) * np.exp(-(dists**2) / (self.L**2))
        return torch.from_numpy(cov).float()

    def _build_H(self, obs_coords):
        """Строит оператор наблюдений H (привязка к ближайшему узлу).
        H имеет размерность [N_obs, N_oi], где N_oi — число OI-узлов
        (ROI или полная сетка).
        """
        num_obs = len(obs_coords)
        num_oi = len(self._oi_coords)
        H = torch.zeros(num_obs, num_oi, device=self.device)

        dists = self._dist_matrix(obs_coords, self._oi_coords)
        closest = np.argmin(dists, axis=1)

        for i, idx in enumerate(closest):
            H[i, idx] = 1.0
        return H

    @torch.no_grad()
    def apply(self, forecast, observations):
        """
        Применяет усвоение.
        forecast: [..., C] (последнее измерение - каналы)
        observations: [..., C] (NaN где нет наблюдений)

        Если roi_idx задан: OI корректирует только ROI-узлы, остальные — без изменений.
        """
        input_shape = forecast.shape

        x_b = forecast.view(-1, input_shape[-1])
        y_o = observations.view(-1, input_shape[-1])

        x_a = x_b.clone()
        G_flat, C = x_b.shape

        if G_flat != len(self.grid_coords):
            raise RuntimeError(f"Размер входного тензора {G_flat} не совпадает "
                               f"с размером сетки ОИ {len(self.grid_coords)}")

        for c in range(C):
            if self.roi_idx is not None:
                # ROI mode: наблюдения и коррекция только внутри ROI
                y_o_roi = y_o[self.roi_idx, c]
                obs_mask_roi = ~torch.isnan(y_o_roi)
                if not obs_mask_roi.any():
                    continue

                y_obs_val = y_o_roi[obs_mask_roi].to(self.device)
                obs_coords = self._oi_coords[obs_mask_roi.cpu().numpy()]

                H = self._build_H(obs_coords)
                R = torch.eye(len(y_obs_val), device=self.device) * (self.sigma_o ** 2)

                x_b_roi = x_b[self.roi_idx, c].to(self.device)  # [N_roi]

                HT = H.T
                BH_T = self.B @ HT
                inv_term = torch.linalg.inv(
                    H @ BH_T + R + torch.eye(len(y_obs_val), device=self.device) * 1e-5
                )
                K = BH_T @ inv_term

                innovation = y_obs_val - (H @ x_b_roi)
                x_a[self.roi_idx, c] = x_b_roi + (K @ innovation)
            else:
                # Full-grid mode (original)
                obs_mask = ~torch.isnan(y_o[:, c])
                if not obs_mask.any():
                    continue

                y_obs_val = y_o[obs_mask, c].to(self.device)
                obs_coords = self.grid_coords[obs_mask.cpu().numpy()]

                H = self._build_H(obs_coords)
                R = torch.eye(len(y_obs_val), device=self.device) * (self.sigma_o ** 2)

                x_b_vec = x_b[:, c].to(self.device)

                HT = H.T
                BH_T = self.B @ HT
                inv_term = torch.linalg.inv(
                    H @ BH_T + R + torch.eye(len(y_obs_val), device=self.device) * 1e-5
                )
                K = BH_T @ inv_term

                innovation = y_obs_val - (H @ x_b_vec)
                x_a[:, c] = x_b_vec + (K @ innovation)

        return x_a.view(input_shape)
