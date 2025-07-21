import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

index_to_class = {0: 'Aji pepper plant', 1: 'Almonds plant', 2: 'Amaranth plant', 3: 'Apples plant', 4: 'Artichoke plant', 5: 'Avocados plant', 6: 'Bananas plant', 7: 'Barley plant', 8: 'Beets plant', 9: 'Black pepper plant', 10: 'Blueberries plant', 11: 'Bok choy plant', 12: 'Brazil nuts plant', 13: 'Broccoli plant', 14: 'Brussels sprout plant', 15: 'Buckwheat plant', 16: 'Cabbages and other brassicas plant', 17: 'Camucamu plant', 18: 'Carrots and turnips plant', 19: 'Cashew nuts plant', 20: 'Cassava plant', 21: 'Cauliflower plant', 22: 'Celery plant', 23: 'Cherimoya plant', 24: 'Cherry plant', 25: 'Chestnuts plant', 26: 'Chickpeas plant', 27: 'Chili peppers and green peppers plant', 28: 'Cinnamon plant', 29: 'Cloves plant', 30: 'Cocoa beans plant', 31: 'Coconuts plant', 32: 'Coffee (green) plant', 33: 'Collards plant', 34: 'Cotton lint plant', 35: 'Cranberries plant', 36: 'Cucumbers and gherkins plant', 37: 'Dates plant', 38: 'Dry beans plant', 39: 'Dry peas plant', 40: 'Durian plant', 41: 'Eggplants (Aubergines) plant', 42: 'Endive plant', 43: 'Fava bean plant', 44: 'Figs plant', 45: 'Flax fiber and tow plant', 46: 'Flaxseed (Linseed) plant', 47: 'Fonio plant', 48: 'Garlic plant', 49: 'Ginger plant', 50: 'Gooseberries plant', 51: 'Grapes plant', 52: 'Groundnuts (Peanuts) plant', 53: 'Guarana plant', 54: 'Guavas plant', 55: 'Habanero pepper plant', 56: 'Hazelnuts plant', 57: 'Hemp plant', 58: 'Hen eggs (shell weight) plant', 59: 'Horseradish plant', 60: 'Jackfruit plant', 61: 'Jute plant', 62: 'Kale plant', 63: 'Kohlrabi plant', 64: 'Leeks plant', 65: 'Lemons and limes plant', 66: 'Lentils plant', 67: 'Lettuce and chicory plant', 68: 'Lima bean plant', 69: 'Longan plant', 70: 'Lupins plant', 71: 'Lychee plant', 72: 'Maize (Corn) plant', 73: 'Mandarins, clementines, satsumas plant', 74: 'Mangoes, mangosteens, guavas plant', 75: 'Maracuja(Passionfruit) plant', 76: 'Millet plant', 77: 'Mint plant', 78: 'Mung bean plant', 79: 'Mustard greens plant', 80: 'Mustard seeds plant', 81: 'Navy bean plant', 82: 'Oats plant', 83: 'Oil palm fruit plant', 84: 'Okra plant', 85: 'Olives plant', 86: 'Onions (dry) plant', 87: 'Oranges plant', 88: 'Oregano plant', 89: 'Papayas plant', 90: 'Parsley plant', 91: 'Peaches and nectarines plant', 92: 'Peas (Green) plant', 93: 'Persimmons plant', 94: 'Pine nuts plant', 95: 'Pineapples plant', 96: 'Pinto bean plant', 97: 'Pistachios plant', 98: 'Plantains plant', 99: 'Pomegranates plant', 100: 'Potatoes plant', 101: 'Pumpkins, squash and gourds plant', 102: 'Quinoa plant', 103: 'Radishes and similar roots plant', 104: 'Rambutan plant', 105: 'Rapeseed (Canola) plant', 106: 'Raspberries plant', 107: 'Rice (Paddy) plant', 108: 'Rosemary plant', 109: 'Rubber (natural) plant', 110: 'Rye plant', 111: 'Saffron plant', 112: 'Sage plant', 113: 'Scallions plant', 114: 'Sorghum plant', 115: 'Soursop plant', 116: 'Soybeans plant', 117: 'Spinach plant', 118: 'Starfruit plant', 119: 'Strawberries plant', 120: 'Sugar beet plant', 121: 'Sugar cane plant', 122: 'Sunflower seeds plant', 123: 'Sweet potatoes plant', 124: 'Swiss chard plant', 125: 'Tamarind plant', 126: 'Taro (cocoyam) plant', 127: 'Tea plant', 128: 'Teff plant', 129: 'Thyme plant', 130: 'Tomatoes plant', 131: 'Triticale plant', 132: 'Turmeric plant', 133: 'Turnip greens plant', 134: 'Vanilla beans plant', 135: 'Walnuts plant', 136: 'Watermelons plant', 137: 'Wheat plant', 138: 'Yams plant'}

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    'train_data/RGB_224x224/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load model
model = load_model('plant_classifier_model.h5')

# Load and preprocess image
img_path = 'train_data/RGB_224x224/val/Yams plant/6422.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_class_index]

# Map class index to label
class_indices = train_data.class_indices

predicted_class = index_to_class[predicted_class_index]

print(f"Predicted plant: {predicted_class} (Confidence: {confidence:.2f})")

