# building_model_Sian_Swin_Unet.py
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# ==========================================
# 1. BLOCOS CUSTOMIZADOS (Com serialização)
# ==========================================

class SwinBlock(layers.Layer):
    """
    Bloco simplificado do Swin Transformer.
    """
    def __init__(self, embed_dim, num_heads, window_size=7, **kwargs):
        super(SwinBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        
        self.mlp = models.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dense(embed_dim)
        ])

    def call(self, x):
        # Atenção
        res = x
        x = self.ln1(x)
        x = self.attn(x, x) 
        x = res + x
        
        # MLP
        res = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = res + x
        return x

    # --- MÉTODO CRUCIAL PARA SALVAR/CARREGAR O MODELO ---
    def get_config(self):
        config = super(SwinBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
        })
        return config

# ==========================================
# 2. FUNÇÕES AUXILIARES DE TOPOLOGIA
# ==========================================

def patch_embed(x, embed_dim, patch_size=4):
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='same')(x)
    x = layers.LayerNormalization()(x)
    return x

def patch_merging(x, filters):
    x = layers.Conv2D(filters, kernel_size=2, strides=2, padding='same')(x)
    return x

def patch_expanding(x, filters):
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
    return x

def encoder_body(input_tensor, filter_list):
    skips = []
    # Stage 1
    x = patch_embed(input_tensor, filter_list[0])
    x = SwinBlock(filter_list[0], num_heads=3)(x)
    skips.append(x)
    # Stage 2
    x = patch_merging(x, filter_list[1])
    x = SwinBlock(filter_list[1], num_heads=6)(x)
    skips.append(x)
    # Stage 3
    x = patch_merging(x, filter_list[2])
    x = SwinBlock(filter_list[2], num_heads=12)(x)
    skips.append(x)
    # Stage 4 (Bottleneck)
    x = patch_merging(x, filter_list[3])
    x = SwinBlock(filter_list[3], num_heads=24)(x)
    return x, skips

def decoder_body(encoded_feature, skips_t1, skips_t2, filter_list):
    x = encoded_feature
    filters = filter_list[::-1]
    
    # Stage 1
    x = patch_expanding(x, filters[1]) 
    s1 = skips_t1[-1]
    s2 = skips_t2[-1]
    diff = layers.Subtract()([s1, s2])
    x = layers.Concatenate()([x, diff]) 
    x = layers.Conv2D(filters[1], 3, padding='same', activation='gelu')(x)
    x = SwinBlock(filters[1], num_heads=12)(x)
    
    # Stage 2
    x = patch_expanding(x, filters[2])
    s1 = skips_t1[-2]
    s2 = skips_t2[-2]
    diff = layers.Subtract()([s1, s2])
    x = layers.Concatenate()([x, diff])
    x = layers.Conv2D(filters[2], 3, padding='same', activation='gelu')(x)
    x = SwinBlock(filters[2], num_heads=6)(x)
    
    # Stage 3
    x = patch_expanding(x, filters[3])
    s1 = skips_t1[-3]
    s2 = skips_t2[-3]
    diff = layers.Subtract()([s1, s2])
    x = layers.Concatenate()([x, diff])
    x = layers.Conv2D(filters[3], 3, padding='same', activation='gelu')(x)
    x = SwinBlock(filters[3], num_heads=3)(x)
    
    return x

# ==========================================
# 3. BUILDER PRINCIPAL
# ==========================================

def build_siam_swin_unet(input_shape=(256, 256, 9), num_classes=1):
    """
    Constrói o Modelo Final Siam-Swin-Unet.
    """
    input_t1 = Input(shape=input_shape, name="Input_T1")
    input_t2 = Input(shape=input_shape, name="Input_T2")
    
    # filters = [96, 192, 384, 768] 
    filters = [64, 128, 256, 512] # um modelo mais pequeno
    
    # Encoder Compartilhado
    enc_input = Input(shape=input_shape)
    enc_out, enc_skips = encoder_body(enc_input, filters)
    encoder_model = models.Model(inputs=enc_input, outputs=[enc_out] + enc_skips, name="Shared_Encoder")
    
    features_t1 = encoder_model(input_t1) 
    features_t2 = encoder_model(input_t2)
    
    bottleneck_t1 = features_t1[0]
    skips_t1 = features_t1[1:]
    
    bottleneck_t2 = features_t2[0]
    skips_t2 = features_t2[1:]
    
    # Fusion Bottleneck
    bottleneck_fusion = layers.Subtract()([bottleneck_t1, bottleneck_t2])
    
    # Decoder
    final_features = decoder_body(bottleneck_fusion, skips_t1, skips_t2, filters)
    
    # Head
    x = layers.Conv2DTranspose(filters[0]//2, kernel_size=4, strides=4, padding='same')(final_features)
    outputs = layers.Conv2D(num_classes, kernel_size=1, activation='sigmoid', name='output_change', dtype='float32')(x)
    
    model = models.Model(inputs=[input_t1, input_t2], outputs=outputs, name="SiamSwinUnet")
    
    return model