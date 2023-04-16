# +
import os
import argparse
import tensorflow as tf
import horovod.tensorflow as hvd
from tqdm import tqdm
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str)
    parser.add_argument(
        "--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")
    )
    parser.add_argument(
        "--epochs", type=int, default=os.environ.get("SM_HP_EPOCHS", 30)
    )

    parser.add_argument(
        "--latent_dim", type=int, default=os.environ.get("SM_HP_LATENT_DIM", 100)
    )
    parser.add_argument(
        "--image_size", type=int, default=os.environ.get("SM_HP_IMAGE_SIZE", 64)
    )
    parser.add_argument(
        "--batch_size", type=int, default=os.environ.get("SM_HP_BATCH_SIZE", 256)
    )

    return parser.parse_known_args()


def prepare_training_data(uri: str, img_size: int, batch_size: int) -> tf.data.Dataset:
    dataset = image_dataset_from_directory(
        directory=uri,
        label_mode=None,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    dataset.map(lambda x: (x - 127.5) / 127.5)
    return dataset


def split_dataset(dataset: tf.data.Dataset, n_gpus: int) -> list[tf.data.Dataset]:
    split_size = int(len(dataset) / n_gpus)
    return [dataset.take(split_size) for _ in range(n_gpus)]


def create_discriminator(img_size: int) -> tf.keras.Sequential:
    discriminator = tf.keras.Sequential(
        [
            layers.Input((img_size, img_size, 3)),
            layers.Conv2D(256, kernel_size=4, padding="same", use_bias=False),
            layers.LeakyReLU(0.2),
            layers.Conv2D(
                512, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            layers.LeakyReLU(0.2),
            layers.Conv2D(
                512, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            layers.LeakyReLU(0.2),
            layers.Conv2D(
                1024, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid", use_bias=False),
        ]
    )

    return discriminator


def create_generator(img_size: int, latent_dim: int) -> tf.keras.Sequential:
    base_size = int(img_size / 16)
    generator = tf.keras.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(base_size * base_size * 512, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Reshape((base_size, base_size, 512)),
            # 4x4x512
            layers.Conv2DTranspose(
                1024, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            # 8x8x1024
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(
                512, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            # 16x16x512
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(
                256, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            # 32x32x256
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(
                3, kernel_size=4, strides=2, padding="same", use_bias=False
            ),
            # 64x64x3
            layers.Activation(tf.keras.activations.tanh),
        ]
    )
    return generator


def train_GAN(
    generator, discriminator, dataset, latent_dim: int, epochs: int, img_output: str
) -> None:

    # adjust learning rate based on number of GPUs.
    opt_gen = tf.keras.optimizers.Adam(1e-4)
    opt_disc = tf.keras.optimizers.Adam(1e-4)
    
    opt_gen = hvd.DistributedOptimizer(opt_gen)
    opt_disc = hvd.DistributedOptimizer(opt_disc)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    for epoch in tqdm(range(epochs)):
        for idx, real in enumerate(dataset):
            batch_size = real.shape[0]
            noise = tf.random.normal((batch_size, latent_dim))

            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                fake = generator(noise, training=True)
                disc_real = discriminator(real, training=True)
                disc_fake = discriminator(fake, training=True)

                loss_disc_real = loss_fn(tf.ones((batch_size, 1)), disc_real)
                loss_disc_fake = loss_fn(tf.zeros((batch_size, 1)), disc_fake)
                loss_disc = loss_disc_real + loss_disc_fake

                loss_gen = loss_fn(tf.ones(batch_size, 1), disc_fake)

                if idx % 100 == 0 and hvd.rank() == 0:
                    img = tf.keras.preprocessing.image.array_to_img(fake[0])
                    out_path = os.path.join(img_output, f"gen_e{epoch}_idx{idx}.png")
                    img.save(out_path)

            disc_tape = hvd.DistributedGradientTape(disc_tape)
            grads = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
            opt_disc.apply_gradients(zip(grads, discriminator.trainable_weights))
            gen_tape = hvd.DistributedGradientTape(gen_tape)
            grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
            opt_gen.apply_gradients(zip(grads, generator.trainable_weights))

            if idx == 0:
                hvd.broadcast_variables(generator.variables, root_rank=0)
                hvd.broadcast_variables(opt_gen.variables(), root_rank=0)
                hvd.broadcast_variables(discriminator.variables, root_rank=0)
                hvd.broadcast_variables(opt_disc.variables(), root_rank=0)


if __name__ == "__main__":
    args, *_ = parse_args()

    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    dataset = prepare_training_data(args.train, args.image_size, args.batch_size)
    datasets = split_dataset(dataset, n_gpus=hvd.size())
    dataset = datasets[hvd.rank()]
    generator = create_generator(args.image_size, args.latent_dim)
    discriminator = create_discriminator(args.image_size)
    train_GAN(
        generator,
        discriminator,
        dataset,
        args.latent_dim,
        args.epochs,
        args.sm_model_dir,
    )
    if hvd.rank() == 0:
        generator.save(os.path.join(args.sm_model_dir, "tf-hvd-cats-generator"))
