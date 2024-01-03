## Convolutional Block

def Convolutional_inc_res_Block(inputs, filters):
    
    shortcut = inputs

    x1 = Conv2D(filters = filters//2, kernel_size = (3,3), padding = 'same', kernel_initializer = 'he_normal')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x1 = Conv2D(filters = filters//2, kernel_size = (3,3), padding = 'same', kernel_initializer = 'he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    """RESIDUAL BLOCK"""

    s = Conv2D(filters = filters/2, kernel_size = (3,3), padding = 'same', kernel_initializer = 'he_normal')(shortcut)
    s = Activation('relu')(s)

    resultant = Concatenate()([x1, s])

    return resultant
  
## Encoder Mechanism

def Encoder_Block_iru(inputs, filters, dropout_rate, pool_size):
    
    conv_output = Convolutional_inc_res_Block(inputs, filters)
    output = MaxPooling2D(pool_size = pool_size)(conv_output)
    output = Dropout(dropout_rate)(output)
    
    return conv_output, output

def Encoder_iru(inputs):
    
    s1, b1 = Encoder_Block_iru(inputs, filters = 64,  dropout_rate = 0.1, pool_size = 2)
    s2, b2 = Encoder_Block_iru(b1,     filters = 128, dropout_rate = 0.1, pool_size = 2)
    s3, b3 = Encoder_Block_iru(b2,     filters = 256, dropout_rate = 0.1, pool_size = 2)
    s4, b4 = Encoder_Block_iru(b3,     filters = 512, dropout_rate = 0.1, pool_size = 2)
    
    return (s1, s2, s3, s4), b4

## Bottleneck Layer

def bottleneck_iru(inputs):
    
    bottle_neck = Convolutional_inc_res_Block(inputs, filters = 1024)
    return bottle_neck

## CBAM Attention Gate

def cbam_attention_gate(skip_X, filters):
    # Feature maps from the encoder path
    encoder_features = skip_X

    # Spatial attention module
    spatial_attention = Conv2D(filters, kernel_size=1, activation='sigmoid', kernel_initializer='he_normal')(encoder_features)
    spatial_attention = Multiply()([encoder_features, spatial_attention])

    # Channel attention module
    channel_attention = tf.reduce_mean(encoder_features, axis=[1, 2], keepdims=True)
    channel_attention = Conv2D(filters, kernel_size=1, activation='sigmoid', kernel_initializer='he_normal')(channel_attention)
    channel_attention = Multiply()([encoder_features, channel_attention])

    # Combine spatial and channel attention
    attention = Add()([spatial_attention, channel_attention])
    attention = Activation('relu')(attention)

    # Apply attention to the encoder features
    weighted_encoder_features = Multiply()([attention, encoder_features])
    weighted_encoder_features = BatchNormalization()(weighted_encoder_features)

    return weighted_encoder_features

## Decoder Mechanism

def Decoder_Block_iru(inputs, conv_output, filters, strides, dropout_rate):
    
    u = Conv2DTranspose(filters, kernel_size = (3,3), strides = strides, padding = 'same')(inputs)
    c = Concatenate()([u, conv_output])
    c = Dropout(dropout_rate)(c)
    c = Convolutional_inc_res_Block(c, filters)
    
    return c

def Decoder_iru(inputs, conv_output, output_channels):

    s1, s2, s3, s4 = conv_output
    
    a1 = cbam_attention_gate(s4, filters = 512)
    d1 = Decoder_Block_iru(inputs,  a1, filters = 512, strides = 2, dropout_rate = 0.2)
    
    a2 = cbam_attention_gate(s3, filters = 256)
    d2 = Decoder_Block_iru(d1,  a2, filters = 256,  strides = 2, dropout_rate = 0.2)
    
    a3 = cbam_attention_gate(s2, filters = 128)
    d3 = Decoder_Block_iru(d2,  a3, filters = 128, strides = 2, dropout_rate = 0.2)

    a4 = cbam_attention_gate(s1, filters = 64)
    d4 = Decoder_Block_iru(d3,  a4, filters = 64, strides = 2, dropout_rate = 0.2)
    
    output = Conv2D(filters = output_channels, kernel_size = (1,1), activation = 'softmax')(d4)
    return output

## Model Instantiation

output_channels = 4
def IR_ATT_UNET():
    
    inputs = Input(shape = (128, 128,2,))
    
    conv_output, encoder_output = Encoder_iru(inputs)
    bottleneck_output = bottleneck_iru(encoder_output)
    decoder_output = Decoder_iru(bottleneck_output, conv_output, output_channels)
    output = Model(inputs = inputs, outputs = decoder_output, name = "IR_ATT_UNET")
    
    return output

model = IR_ATT_UNET()
model.summary()


