import pickle

def save_model_scaler(model, scaler, model_path='model.pkl',
                      scaler_path='scaler.pkl'):
    
    
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
        
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)



def load_model_scaler(model_path='model.pkl', scaler_path='scaler.pkl'):
    
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    
    return model, scaler
    