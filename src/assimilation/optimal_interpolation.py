import torch
import numpy as np

class OptimalInterpolation:
    def __init__(self, grid_lats, grid_lons, sigma_b, sigma_o, L, device):
        """
        Инициализация Оптимальной Интерполяции.
        sigma_b: ошибка прогноза
        sigma_o: ошибка наблюдений
        L: радиус влияния (в метрах)
        """
        self.sigma_b = sigma_b
        self.sigma_o = sigma_o
        self.L = L
        self.device = device

        # Создаем сетку координат (широта, долгота) для всех точек
        lat_grid, lon_grid = np.meshgrid(grid_lats, grid_lons, indexing='ij')
        self.grid_coords = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
        
        # Сразу считаем матрицу ошибок модели B (это долго, но делается 1 раз)
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
        dists = self._dist_matrix(self.grid_coords, self.grid_coords)
        cov = (self.sigma_b**2) * np.exp(-(dists**2) / (self.L**2))
        return torch.from_numpy(cov).float()

    def _build_H(self, obs_coords):
        """Строит оператор наблюдений H (привязка к ближайшему узлу)."""
        num_obs = len(obs_coords)
        num_grid = len(self.grid_coords)
        H = torch.zeros(num_obs, num_grid, device=self.device)
        
        # Ищем ближайший узел сетки для каждого наблюдения
        dists = self._dist_matrix(obs_coords, self.grid_coords)
        closest = np.argmin(dists, axis=1)
        
        for i, idx in enumerate(closest):
            H[i, idx] = 1.0
        return H

    @torch.no_grad()
    def apply(self, forecast, observations):
        """
        Применяет усвоение.
        forecast: [..., C] (последнее измерение - каналы)
        observations: [..., C]
        """
        # Сохраняем исходную форму
        input_shape = forecast.shape
        
        # Превращаем в плоский вид [Total_Nodes, C]
        # Если вход [G, C] -> остается [G, C]
        # Если вход [1, G, C] -> становится [G, C]
        x_b = forecast.view(-1, input_shape[-1])
        y_o = observations.view(-1, input_shape[-1])
        
        x_a = x_b.clone()
        G_flat, C = x_b.shape # Теперь G_flat это G или 1*G

        for c in range(C):
            # Берем данные только там, где они есть (не NaN)
            obs_mask = ~torch.isnan(y_o[:, c])
            if not obs_mask.any(): continue

            # Координаты и значения наблюдений
            y_obs_val = y_o[obs_mask, c].to(self.device)
            
            # ВАЖНО: obs_mask может быть длиннее, чем grid_coords, если у нас батч > 1
            # Но в нашем случае мы ожидаем, что apply вызывается для одного поля [G, C]
            # Если G_flat != len(grid_coords), значит что-то не так с логикой вызова
            if G_flat != len(self.grid_coords):
                 # Пытаемся понять, это батч или ошибка
                 raise RuntimeError(f"Размер входного тензора {G_flat} не совпадает с размером сетки ОИ {len(self.grid_coords)}")

            obs_coords = self.grid_coords[obs_mask.cpu().numpy()]
            
            # Матрицы H и R
            H = self._build_H(obs_coords)
            R = torch.eye(len(y_obs_val), device=self.device) * (self.sigma_o**2)
            
            # Основная формула ОИ
            x_b_vec = x_b[:, c].to(self.device) # [G]
            
            HT = H.T
            BH_T = self.B @ HT
            # Добавляем регуляризацию, чтобы не падало
            inv_term = torch.linalg.inv(H @ BH_T + R + torch.eye(len(y_obs_val), device=self.device) * 1e-5)
            K = BH_T @ inv_term
            
            # Поправка
            innovation = y_obs_val - (H @ x_b_vec)
            x_a[:, c] = x_b_vec + (K @ innovation)

        return x_a.view(input_shape)
