import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib

class CreditScoringModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier()
        }
        self.best_models = {}
        self.results = {}
        
        plt.style.use('ggplot')
        sns.set_palette("husl")
        
    def load_data(self):
        print("\n=== Cargando y preparando datos ===")
        self.df = pd.read_csv(self.file_path, sep=',', decimal='.')
        self.df.columns = self.df.columns.str.strip()
        self.df['tipo'] = self.df['tipo'].replace({'Otro': 'Nuevo', 'Nuevo': 'Nuevo', 'Renovado': 'Renovado'})
        print(f"Datos cargados: {self.df.shape[0]} registros, {self.df.shape[1]} columnas")
        return self
    
    def eda(self):
        print("\n=== Realizando análisis exploratorio ===")
        
        nuevos = self.df[self.df['tipo'] == 'Nuevo']
        renovados = self.df[self.df['tipo'] == 'Renovado']
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.countplot(data=nuevos, x='atraso_30', ax=ax[0])
        ax[0].set_title('Distribución de Default - Nuevos')
        sns.countplot(data=renovados, x='atraso_30', ax=ax[1])
        ax[1].set_title('Distribución de Default - Renovados')
        plt.show()
        
        print("\nCorrelaciones con target (Nuevos):")
        print(nuevos.corr(numeric_only=True)['atraso_30'].sort_values(ascending=False).head(5))
        print("\nCorrelaciones con target (Renovados):")
        print(renovados.corr(numeric_only=True)['atraso_30'].sort_values(ascending=False).head(5))
        
        return self
    
    def train_models(self):
        print("\n=== Entrenando modelos ===")
        
        for grupo in ['Nuevo', 'Renovado']:
            print(f"\n--- Entrenando para grupo: {grupo} ---")
            df_group = self.df[self.df['tipo'] == grupo]
            
            # Selección de variables según tipo
            base_features = [
                'edad', 'monto_solicitado', 'cant_cuotas', 'ingreso',
                'antiguedad_laboral', 'sexo', 'estado_civil', 'tipo_sucursal',
                'aporta_ips', 'aporta_iva'
            ]
            
            if grupo == 'Renovado':
                features = base_features + ['promedio_atraso_negofin', 'maximo_atraso_negofin']
            else:
                features = base_features
                
            X = df_group[features]
            y = df_group['atraso_30']
            
            # Configuración dinámica del preprocesador
            numeric_features = [f for f in features if f not in ['sexo', 'estado_civil', 'tipo_sucursal', 'aporta_iva', 'aporta_ips']]
            categorical_features = ['sexo', 'estado_civil', 'tipo_sucursal', 'aporta_iva', 'aporta_ips']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            resultados = {}
            best_auc = 0
            best_model = None
            
            for name, model in self.models.items():
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                pipeline.fit(X_train, y_train)
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                resultados[name] = auc
                
                if auc > best_auc:
                    best_auc = auc
                    best_model = pipeline
                    
                print(f"{name}: AUC = {auc:.4f}")
            
            self.results[grupo] = resultados
            self.best_models[grupo] = (best_model, features)
            print(f"\nMejor modelo para {grupo}: {best_model.named_steps['model'].__class__.__name__} (AUC = {best_auc:.4f})")
        
        return self
    
    def save_models(self):
        print("\n=== Guardando modelos ===")
        for grupo, (modelo, features) in self.best_models.items():
            filename = f"mejor_modelo_micro_{grupo.lower()}.pkl"
            joblib.dump((modelo, features), filename)
            print(f"Modelo guardado como: {filename} (con {len(features)} features)")
        return self
    
    def show_results(self):
        print("\n=== Resultados Finales ===")
        resultados_df = pd.DataFrame(self.results).T
        print(resultados_df)
        return self

if __name__ == "__main__":
    pipeline = (
        CreditScoringModel('bd_micro_entrenamiento.csv')
        .load_data()
        .eda()
        .train_models()
        .show_results()
        .save_models()
    )