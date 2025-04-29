import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from lightgbm import LGBMClassifier
import joblib
import os

# Reemplazar el pipeline de sklearn con el pipeline de imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

class EnhancedCreditScoringModel:
    """
    Modelo avanzado de evaluación crediticia que utiliza LightGBM y optimización
    de hiperparámetros para predecir el riesgo de incumplimiento en préstamos.
    
    Esta clase maneja el flujo completo desde la carga de datos hasta el entrenamiento
    y evaluación de modelos para clientes nuevos y renovados.
    """
    
    def __init__(self, file_path):
        """
        Inicializa el modelo con la ruta del archivo de datos.
        
        Args:
            file_path (str): Ruta al archivo CSV con los datos de préstamos
        """
        self.file_path = file_path
        self.df = None
        self.best_models = {}  # Almacena los mejores modelos entrenados
        self.results = {}      # Almacena los resultados de evaluación
        self.studies = {}      # Almacena los estudios de Optuna
        self.class_distributions = {}  # Almacena distribución de clases por grupo
        
        # Configuración de estilo para visualizaciones
        plt.style.use('ggplot')
        sns.set_palette("husl")
        shap.initjs()

    def load_data(self):
        """
        Carga los datos desde el archivo CSV y realiza una limpieza inicial.
        Estandariza la columna 'tipo' para tener solo 'Nuevo' y 'Renovado'.
        
        Returns:
            self: Para permitir encadenamiento de métodos
        """
        print("\n=== Cargando y preparando datos ===")
        self.df = pd.read_csv(self.file_path, sep=',', decimal='.')
        # Estandariza la categoría 'tipo' para que solo tenga 'Nuevo' o 'Renovado'
        self.df['tipo'] = self.df['tipo'].replace({'Otro': 'Nuevo', 'Nuevo': 'Nuevo', 'Renovado': 'Renovado'})
        print(f"Datos cargados: {self.df.shape[0]} registros")
        
        # Analizar la distribución de clases para cada grupo
        for grupo in ['Nuevo', 'Renovado']:
            df_grupo = self.df[self.df['tipo'] == grupo]
            class_counts = df_grupo['atraso_30'].value_counts()
            ratio = class_counts.get(1, 0) / class_counts.get(0, 1)
            self.class_distributions[grupo] = {
                'count_class_0': class_counts.get(0, 0),
                'count_class_1': class_counts.get(1, 0),
                'ratio': ratio
            }
            print(f"\nDistribución para {grupo}:")
            print(f"  - Clase 0 (no default): {class_counts.get(0, 0)}")
            print(f"  - Clase 1 (default): {class_counts.get(1, 0)}")
            print(f"  - Ratio (minoritaria/mayoritaria): {ratio:.4f}")
        
        return self

    def feature_engineering(self):
        """
        Crea nuevas características a partir de los datos existentes para
        mejorar el poder predictivo del modelo.
        
        Returns:
            self: Para permitir encadenamiento de métodos
        """
        print("\n=== Ingeniería de características ===")
        
        # Ratio entre monto solicitado e ingreso (capacidad de pago)
        self.df['ratio_pago'] = self.df['monto_solicitado'] / (self.df['ingreso'] + 1e-6)
        
        # Categorización de la edad en rangos
        self.df['edad_bin'] = pd.cut(
            self.df['edad'], 
            bins=[20, 30, 40, 50, 60, 100], 
            labels=['20-30', '30-40', '40-50', '50-60', '60+']
        )
        
        # Factor de riesgo histórico (solo para clientes renovados)
        self.df['riesgo_historico'] = np.where(
            self.df['tipo'] == 'Renovado',
            self.df['promedio_atraso_negofin'] * self.df['maximo_atraso_negofin'],
            0
        )
        
        # Añadir nuevas características más avanzadas
        
        # Ratio de antigüedad laboral con respecto a la edad
        self.df['ratio_antiguedad_edad'] = self.df['antiguedad_laboral'] / (self.df['edad'] + 1e-6)
        
        # Carga de deuda ajustada por cuotas
        self.df['carga_por_cuota'] = self.df['monto_solicitado'] / (self.df['cant_cuotas'] + 1e-6)
        
        print("Nuevas características creadas:")
        print("- ratio_pago: Monto solicitado / Ingreso")
        print("- edad_bin: Rangos de edad categorizados")
        print("- riesgo_historico: (Solo renovados) Promedio * Máximo atraso")
        print("- ratio_antiguedad_edad: Antigüedad laboral / Edad")
        print("- carga_por_cuota: Monto solicitado / Cantidad de cuotas")
        return self

    def eda(self):
        """
        Realiza un análisis exploratorio de datos (EDA) para visualizar
        las correlaciones entre variables y el target (atraso_30).
        
        Returns:
            self: Para permitir encadenamiento de métodos
        """
        print("\n=== Análisis exploratorio ===")

        # Crear carpeta si no existe
        os.makedirs("eda", exist_ok=True)

        
        # VVisualiza correlaciones con atraso_30 por tipo de cliente
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        for i, grupo in enumerate(['Nuevo', 'Renovado']):
            df_g = self.df[self.df['tipo'] == grupo]
            corr = df_g.corr(numeric_only=True)['atraso_30'].sort_values(ascending=False)
            sns.barplot(x=corr.values, y=corr.index, ax=ax[i])
            ax[i].set_title(f'Correlación con Default - {grupo}')
        plt.tight_layout()
        plt.savefig(os.path.join("eda", "correlaciones.png"))  # Guardar figura
        
        # Añadir visualización de distribución de características principales
        nums = ['edad', 'ingreso', 'monto_solicitado', 'ratio_pago', 'carga_por_cuota']
        fig, axs = plt.subplots(len(nums), 2, figsize=(16, 4*len(nums)))
        
        for i, var in enumerate(nums):
            for j, grupo in enumerate(['Nuevo', 'Renovado']):
                df_g = self.df[self.df['tipo'] == grupo]
                sns.histplot(data=df_g, x=var, hue='atraso_30', bins=30, 
                           kde=True, ax=axs[i, j])
                axs[i, j].set_title(f'{var} - {grupo}')
        
        plt.tight_layout()
        plt.savefig(os.path.join("eda", "distribuciones_variables.png"))  # Guardar figura
        
        return self

    def create_pipeline(self, grupo, trial=None):
        """
        Crea un pipeline de procesamiento y modelado específico para cada grupo
        de clientes (Nuevo o Renovado).
        
        Args:
            grupo (str): Tipo de cliente ('Nuevo' o 'Renovado')
            trial (optuna.Trial, optional): Trial de Optuna para optimización
            
        Returns:
            tuple: (Pipeline configurado, lista de características utilizadas)
        """
        # Características base comunes para ambos tipos de clientes
        base_features = [
            'edad', 'monto_solicitado', 'cant_cuotas', 'ingreso',
            'antiguedad_laboral', 'sexo', 'estado_civil', 'tipo_sucursal',
            'ratio_pago', 'edad_bin', 'ratio_antiguedad_edad', 'carga_por_cuota'
        ]
        
        # Para clientes renovados, añadir características de historial
        if grupo == 'Renovado':
            base_features += ['riesgo_historico', 'promedio_atraso_negofin', 'maximo_atraso_negofin']

        # Separar características numéricas y categóricas
        numeric_features = [f for f in base_features if f not in ['sexo', 'estado_civil', 'tipo_sucursal', 'edad_bin']]
        categorical_features = ['sexo', 'estado_civil', 'tipo_sucursal', 'edad_bin']

        # Crear preprocesador para transformar los datos
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),  # Escala las variables numéricas
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # Codifica variables categóricas
        ])

        # Calcular el peso para balancear clases desbalanceadas
        scale_pos_weight = self.df[self.df['tipo'] == grupo]['atraso_30'].value_counts().get(0, 1) / \
                         self.df[self.df['tipo'] == grupo]['atraso_30'].value_counts().get(1, 1)
        
        # Determinar la estrategia adecuada para SMOTE
        current_ratio = self.class_distributions[grupo]['ratio']
        smote_strategy = 'auto'  # Esto usa 1.0 como target ratio
        
        # Configurar hiperparámetros para LightGBM
        base_params = {
            'scale_pos_weight': scale_pos_weight,
            'metric': 'auc',
            'verbose': -1
        }
        
        # Parámetros específicos para optimización LightGBM
        if trial is not None:
            base_params.update({
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            })
        
        classifier = LGBMClassifier(**base_params)
        
        # Crear pipeline basado en las distribuciones de datos
        steps = [
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ]
        
        # Solo añadir SMOTE si hay suficientes muestras en la clase minoritaria
        # y si la clase está suficientemente desbalanceada
        min_class_samples = self.class_distributions[grupo]['count_class_1']
        if min_class_samples >= 5 and current_ratio < 0.8:
            steps.insert(1, ('sampling', SMOTE(sampling_strategy=smote_strategy, random_state=42)))
            print(f"Aplicando SMOTE para {grupo} con strategy={smote_strategy}")
        else:
            print(f"Omitiendo SMOTE para {grupo} debido a las distribuciones de clase")
        
        # Crear y devolver el pipeline completo usando imblearn.pipeline.Pipeline
        pipeline = Pipeline(steps)
        
        return pipeline, base_features

    def objective(self, trial, X, y, grupo):
        """
        Función objetivo para optimización de hiperparámetros con Optuna.
        Evalúa un conjunto de hiperparámetros utilizando validación cruzada.
        
        Args:
            trial: Objeto trial de Optuna
            X: Features de entrenamiento
            y: Target variable
            grupo: Tipo de cliente ('Nuevo' o 'Renovado')
            
        Returns:
            float: Puntuación AUC promedio de validación cruzada
        """
        # Configurar validación cruzada estratificada
        cv = StratifiedKFold(n_splits=3)
        scores = []
        
        # Realizar validación cruzada
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Crear modelo con hiperparámetros del trial
            model, _ = self.create_pipeline(grupo, trial)
            
            # Entrenar y evaluar
            model.fit(X_train, y_train)
            
            # Obtener predicciones directamente del modelo
            y_pred = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_pred))
            
        # Devolver el promedio de las puntuaciones AUC
        return np.mean(scores)

    def train_optimized_model(self, grupo):
        """
        Entrena un modelo optimizado para un grupo específico de clientes
        y genera visualizaciones SHAP para interpretabilidad.
        
        Args:
            grupo (str): Tipo de cliente ('Nuevo' o 'Renovado')
            
        Returns:
            tuple: (Modelo entrenado, características utilizadas, mejor valor AUC)
        """
        print(f"\n--- Optimización de LightGBM para {grupo} ---")
        # Filtrar datos para el grupo específico
        df_group = self.df[self.df['tipo'] == grupo]
        # Obtener lista de características
        _, features = self.create_pipeline(grupo)
        X = df_group[features]
        y = df_group['atraso_30']

        # Crear y ejecutar estudio de optimización
        study_name = f"lightgbm_{grupo}"
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, X, y, grupo), 
            n_trials=30, 
            show_progress_bar=True
        )
        
        # Almacenar el estudio para referencia
        self.studies[study_name] = study

        # Crear un modelo final con los mejores parámetros
        print(f"Mejores parámetros para LightGBM - {grupo}:")
        for param_name, param_value in study.best_params.items():
            print(f"  - {param_name}: {param_value}")
        
        # Crear un nuevo trial con los mejores parámetros para construir el modelo final
        best_trial = optuna.trial.FixedTrial(study.best_params)
        final_model, _ = self.create_pipeline(grupo, best_trial)
        
        # Entrenar el modelo final con todos los datos
        final_model.fit(X, y)

        # Generar visualizaciones SHAP para interpretabilidad
        # Extraer el modelo directamente
        model_instance = final_model.named_steps['classifier']
        
        # Obtener datos preprocesados para SHAP
        if 'preprocessor' in final_model.named_steps:
            preprocessor = final_model.named_steps['preprocessor']
            X_preprocessed = preprocessor.transform(X)
            feature_names = preprocessor.get_feature_names_out()
            X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)
        else:
            # Si por alguna razón no hay preprocesador, usar los datos originales
            X_preprocessed_df = X
        
        # Crear el explicador SHAP
        explainer = shap.TreeExplainer(model_instance)
        shap_values = explainer.shap_values(X_preprocessed_df)
        
        # Manejar los valores SHAP para LightGBM
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Para el caso binario, usar valores para la clase positiva
        
        # Visualizar importancia de características
        plt.figure(figsize=(12, 6))
        
        shap.summary_plot(
            shap_values, 
            X_preprocessed_df,
            feature_names=feature_names if 'preprocessor' in final_model.named_steps else X.columns.tolist(),
            show=False
        )
        plt.title(f'Importancia de variables - LightGBM - {grupo}')
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join("eda", f'Importancia de variables - LightGBM - {grupo}'))  # Guardar figura

        return final_model, features, study.best_value

    def train_all_models(self):
        """
        Entrena modelos optimizados para cada grupo de clientes
        (Nuevo y Renovado) y almacena los resultados.
        
        Returns:
            self: Para permitir encadenamiento de métodos
        """
        print("\n=== Entrenamiento de modelos optimizados ===")
        
        # Entrenar modelos para cada grupo
        for grupo in ['Nuevo', 'Renovado']:
            # Entrenar LightGBM
            model, features, auc = self.train_optimized_model(grupo)
            
            # Calibrar el modelo
            print(f"\n=== Calibrando el modelo LightGBM para {grupo} ===")
            calibrated_model, calibrated_features = self.calibrate_model(grupo)
            
            # Guardar el modelo calibrado para este grupo
            self.best_models[grupo] = (
                calibrated_model,
                calibrated_features
            )
            
            # Guardar resultados
            self.results[grupo] = {
                'auc': auc
            }
            
            print(f"\nResultados para {grupo}:")
            print(f"  - AUC: {auc:.4f}")
        
        return self
    
    def calibrate_model(self, grupo, cv=5):
        """
        Calibra un modelo específico para un grupo de clientes utilizando CalibratedClassifierCV.
        
        Args:
            grupo (str): Tipo de cliente ('Nuevo' o 'Renovado')
            cv (int): Número de folds para la validación cruzada
            
        Returns:
            tuple: (Modelo calibrado, características utilizadas)
        """
        print(f"\n=== Calibración de modelo LightGBM para {grupo} ===")
        
        # Filtrar datos para el grupo específico
        df_group = self.df[self.df['tipo'] == grupo]
        
        # Obtener lista de características
        _, features = self.create_pipeline(grupo)
        X = df_group[features]
        y = df_group['atraso_30']
        
        # Crear el modelo base con los mejores parámetros
        study_name = f"lightgbm_{grupo}"
        if study_name not in self.studies:
            raise ValueError(f"No hay estudio de optimización para LightGBM en {grupo}")
            
        best_trial = optuna.trial.FixedTrial(self.studies[study_name].best_params)
        base_model, _ = self.create_pipeline(grupo, best_trial)
        
        # Entrenar el modelo base
        base_model.fit(X, y)
        
        # Extraer el clasificador del pipeline
        classifier = base_model.named_steps['classifier']
        
        # Crear y entrenar el modelo calibrado
        calibrated_model = CalibratedClassifierCV(
            estimator=classifier,
            cv=cv,
            method='sigmoid',  # Platt scaling
            n_jobs=-1
        )
        
        # Obtener datos preprocesados para la calibración
        preprocessor = base_model.named_steps['preprocessor']
        X_preprocessed = preprocessor.transform(X)
        
        # Obtener nombres de características después del preprocesamiento
        feature_names = preprocessor.get_feature_names_out()
        
        # Convertir a DataFrame para preservar los nombres de características
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)
        
        # Entrenar el modelo calibrado
        calibrated_model.fit(X_preprocessed_df, y)
        
        # Crear un nuevo pipeline con el modelo calibrado
        calibrated_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', calibrated_model)
        ])
        
        # Evaluar la calibración
        self.evaluate_calibration(calibrated_pipeline, X, y, grupo)
        
        return calibrated_pipeline, features
        
    def evaluate_calibration(self, model, X, y, grupo):
        """
        Evalúa la calibración de un modelo utilizando curvas de calibración
        y métricas como Brier Score.
        
        Args:
            model: Modelo a evaluar
            X: Features de entrada
            y: Target variable
            grupo (str): Tipo de cliente ('Nuevo' o 'Renovado')
            
        Returns:
            dict: Métricas de calibración
        """
        # Obtener probabilidades predichas
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calcular curva de calibración
        prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
        
        # Calcular Brier Score
        brier = brier_score_loss(y, y_pred_proba)
        
        # Visualizar curva de calibración
        plt.figure(figsize=(10, 6))
        plt.plot(prob_pred, prob_true, 's-', label=f'LightGBM - {grupo}')
        plt.plot([0, 1], [0, 1], '--', label='Calibración perfecta')
        plt.xlabel('Probabilidad predicha')
        plt.ylabel('Probabilidad real')
        plt.title(f'Curva de Calibración - LightGBM - {grupo}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("eda", f'Calibración - LightGBM - {grupo}.png'))
        
        # Calcular métricas adicionales
        calibration_metrics = {
            'brier_score': brier,
            'mean_abs_error': np.mean(np.abs(prob_true - prob_pred)),
            'max_abs_error': np.max(np.abs(prob_true - prob_pred))
        }
        
        print(f"\nMétricas de calibración para LightGBM - {grupo}:")
        print(f"  - Brier Score: {calibration_metrics['brier_score']:.4f}")
        print(f"  - Error absoluto medio: {calibration_metrics['mean_abs_error']:.4f}")
        print(f"  - Error absoluto máximo: {calibration_metrics['max_abs_error']:.4f}")
        
        # Determinar si la calibración es válida
        # Un Brier Score < 0.25 generalmente se considera bueno para problemas binarios
        is_valid = calibration_metrics['brier_score'] < 0.25 and calibration_metrics['mean_abs_error'] < 0.1
        print(f"  - Calibración válida: {'Sí' if is_valid else 'No'}")
        
        return calibration_metrics

    def save_models(self):
        """
        Guarda los mejores modelos optimizados en archivos para uso posterior.
        
        Returns:
            self: Para permitir encadenamiento de métodos
        """
        print("\n=== Guardando modelos ===")
        # Guardar todos los modelos entrenados
        for grupo in ['Nuevo', 'Renovado']:
            study_name = f"lightgbm_micro_{grupo}"
            if study_name in self.studies:
                model, features = self.create_pipeline(
                    grupo, 
                    optuna.trial.FixedTrial(self.studies[study_name].best_params)
                )
                
                # Reentrenar el modelo con todos los datos
                df_group = self.df[self.df['tipo'] == grupo]
                X = df_group[features]
                y = df_group['atraso_30']
                model.fit(X, y)
                
                # Guardar el modelo
                filename = f"modelo_lightgbm_{grupo.lower()}.pkl"
                joblib.dump({'model': model, 'features': features}, filename)
                print(f"Modelo LightGBM para {grupo} guardado")
        
        # Guardar el mejor modelo de cada grupo (ahora calibrado)
        for grupo, (model, features) in self.best_models.items():
            # Evaluar la calibración para guardar las métricas
            df_group = self.df[self.df['tipo'] == grupo]
            X = df_group[features]
            y = df_group['atraso_30']
            calibration_metrics = self.evaluate_calibration(model, X, y, grupo)
            
            filename = f"mejor_modelo_micro_{grupo.lower()}.pkl"
            joblib.dump({
                'model': model, 
                'features': features, 
                'tipo': 'lightgbm',
                'metricas': self.results[grupo],
                'calibracion': calibration_metrics
            }, filename)
            print(f"Mejor modelo para {grupo} (LightGBM) guardado con métricas de calibración")
            
        return self
    
    def predict_proba(self, data, grupo):
        """
        Realiza predicciones de probabilidad usando el mejor modelo
        para un grupo específico.
        
        Args:
            data (pd.DataFrame): Datos para predecir
            grupo (str): Tipo de cliente ('Nuevo' o 'Renovado')
            
        Returns:
            np.array: Probabilidades predichas
        """
        if grupo not in self.best_models:
            raise ValueError(f"No hay modelo entrenado para el grupo '{grupo}'")
            
        model, features = self.best_models[grupo]
        
        # Verificar que todas las características necesarias estén presentes
        missing_features = set(features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Faltan características en los datos: {missing_features}")
        
        # Extraer solo las características necesarias
        X = data[features]
        
        # Obtener datos preprocesados para la predicción
        if 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            X_preprocessed = preprocessor.transform(X)
            feature_names = preprocessor.get_feature_names_out()
            X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)
        else:
            X_preprocessed_df = X
            
        # Realizar predicción
        return model.predict_proba(X_preprocessed_df)[:, 1]

if __name__ == "__main__":
    # Configurar y ejecutar el pipeline completo de modelado
    pipeline = (
        EnhancedCreditScoringModel('bd_micro_entrenamiento.csv')
        .load_data()
        .feature_engineering()
        .eda()
        .train_all_models()
        .save_models()
    )