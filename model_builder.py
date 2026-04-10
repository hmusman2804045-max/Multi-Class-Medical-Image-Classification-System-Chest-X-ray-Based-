import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def build_resnet_model(input_shape=(224, 224, 3), num_classes=4, fine_tune_layers=None):
    base_model = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False 
    
    if fine_tune_layers is not None and fine_tune_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
            
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name="Global_Avg_Pool"),
        layers.Dropout(0.5, name="Dropout_1"),
        layers.Dense(256, activation='relu', name="Dense_1"),
        layers.Dropout(0.3, name="Dropout_2"),
        layers.Dense(num_classes, activation='softmax', name="Output_Classification")
    ], name="Medical_ResNet50")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

if __name__ == "__main__":
    model = build_resnet_model()
    model.summary()
